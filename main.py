from fastapi import FastAPI, HTTPException, Depends, status, UploadFile, File
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
from pydantic import BaseModel, Field, ConfigDict, EmailStr
from typing import List, Optional, Annotated, Any, Dict
from datetime import datetime, timezone, timedelta
from bson import ObjectId
import os
from jose import JWTError, jwt
import bcrypt
from contextlib import asynccontextmanager
from dotenv import load_dotenv
from pydantic.functional_validators import BeforeValidator
from typing_extensions import TypedDict
import csv
import io
from collections import defaultdict

# Load environment variables
load_dotenv()

# TV White Space frequency bands (typical for Nigeria)
# Channel number -> center frequency in MHz
TVWS_CHANNELS = {
    # VHF Low Band (channels 2-6)
    2: 54, 3: 61, 4: 68, 5: 79, 6: 88,
    # VHF High Band (channels 7-13)
    7: 177, 8: 183, 9: 189, 10: 195, 11: 201, 12: 207, 13: 213,
    # UHF Band (channels 14-51)
    14: 470, 15: 476, 16: 482, 17: 488, 18: 494, 19: 500,
    20: 506, 21: 512, 22: 518, 23: 524, 24: 530, 25: 536,
    26: 542, 27: 548, 28: 554, 29: 560, 30: 566, 31: 572,
    32: 578, 33: 584, 34: 590, 35: 596, 36: 602, 37: 608,
    38: 614, 39: 620, 40: 626, 41: 632, 42: 638, 43: 644,
    44: 650, 45: 656, 46: 662, 47: 668, 48: 674, 49: 680,
    50: 686, 51: 692
}

# Each TV channel is 8 MHz wide; center ± 4 MHz defines the channel boundary.
# Readings from the RF Explorer are spaced ~0.9 MHz apart, so multiple sub-channel
# readings fall within one 8 MHz TV channel.  We bin them and aggregate.
CHANNEL_HALF_BW_MHZ = 4.0

# Signal threshold for TVWS availability (ITU / NCC Nigeria guidelines)
TVWS_FREE_THRESHOLD_DBM = -97.0


def get_channel_from_frequency(freq_mhz: float) -> int:
    """
    Return the TV channel number whose 8 MHz slot contains freq_mhz.
    Returns 0 if the frequency does not fall in any defined TVWS channel.
    """
    for channel, center in TVWS_CHANNELS.items():
        if abs(center - freq_mhz) <= CHANNEL_HALF_BW_MHZ:
            return channel
    return 0


def get_tvws_frequencies() -> List[float]:
    """Return the list of TVWS channel center frequencies (MHz)."""
    return list(TVWS_CHANNELS.values())


def aggregate_readings_to_channels(
    raw_readings: List[Dict]
) -> List[Dict]:
    """
    Aggregate multiple sub-channel RF readings into per-TV-channel summaries.

    The RF Explorer sweeps at ~0.9 MHz steps, so each 8 MHz TV channel contains
    roughly 8-9 individual readings.  We:
      1. Bin each reading into its TV channel.
      2. Take the *maximum* power within the channel (worst-case occupancy).
      3. Mark the channel 'occupied' if max power >= TVWS_FREE_THRESHOLD_DBM,
         else 'free'.

    Args:
        raw_readings: list of dicts with keys
            {channel, frequency_mhz, signal_strength_dbm}

    Returns:
        List of dicts (one per channel) sorted by channel number.
    """
    channel_buckets: Dict[int, List[float]] = defaultdict(list)
    channel_center: Dict[int, float] = {}

    for r in raw_readings:
        ch = r["channel"]
        if ch == 0:
            continue  # skip readings that didn't map to a channel
        channel_buckets[ch].append(r["signal_strength_dbm"])
        channel_center[ch] = TVWS_CHANNELS.get(ch, r["frequency_mhz"])

    aggregated = []
    for ch in sorted(channel_buckets.keys()):
        powers = channel_buckets[ch]
        max_power = max(powers)
        aggregated.append({
            "channel": ch,
            "frequency_mhz": channel_center[ch],
            "signal_strength_dbm": round(max_power, 2),
            "status": "free" if max_power < TVWS_FREE_THRESHOLD_DBM else "occupied",
        })

    return aggregated


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

def validate_object_id(v: Any) -> str:
    if isinstance(v, ObjectId):
        return str(v)
    if isinstance(v, str):
        if ObjectId.is_valid(v):
            return v
    raise ValueError("Invalid ObjectId")

PyObjectId = Annotated[str, BeforeValidator(validate_object_id)]

class MongoBaseModel(BaseModel):
    id: PyObjectId = Field(alias="_id", default=None)
    model_config = ConfigDict(
        populate_by_name=True,
        arbitrary_types_allowed=True,
        json_encoders={ObjectId: str},
    )

    @classmethod
    def from_mongo(cls, data: dict):
        if not data:
            return data
        data['id'] = str(data.pop('_id')) if '_id' in data else None
        return cls(**data)

class User(MongoBaseModel):
    email: EmailStr
    password_hash: str
    role: str
    name: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

class UserCreate(BaseModel):
    email: EmailStr
    password: str
    role: str = "user"
    name: str

class UserUpdate(BaseModel):
    email: Optional[EmailStr] = None
    password: Optional[str] = None
    role: Optional[str] = None
    name: Optional[str] = None

class UserLogin(BaseModel):
    email: EmailStr
    password: str

class UserResponse(BaseModel):
    id: str
    email: EmailStr
    role: str
    name: str
    created_at: datetime
    updated_at: datetime

    @classmethod
    def from_mongo(cls, data: dict):
        if not data:
            return data
        id = str(data.pop('_id'))
        return cls(id=id, **data)

class State(MongoBaseModel):
    name: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

class StateCreate(BaseModel):
    name: str

class StateUpdate(BaseModel):
    name: Optional[str] = None

class Coordinates(TypedDict):
    lat: float
    lon: float

class Location(MongoBaseModel):
    state: str
    name: str
    coordinates: Coordinates
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

class LocationCreate(BaseModel):
    state: str
    name: str
    coordinates: Coordinates

class LocationUpdate(BaseModel):
    state: Optional[str] = None
    name: Optional[str] = None
    coordinates: Optional[Coordinates] = None

class ChannelReading(BaseModel):
    channel: int
    frequency_mhz: float
    signal_strength_dbm: float
    status: Optional[str] = None

class Measurement(MongoBaseModel):
    state: str
    location: str
    timestamp: datetime
    readings: List[ChannelReading]
    created_at: datetime = Field(default_factory=datetime.utcnow)
    source_file: Optional[str] = None
    file_segment: Optional[int] = None
    rbw_khz: Optional[float] = None

class MeasurementCreate(BaseModel):
    state: str
    location: str
    timestamp: datetime
    readings: List[ChannelReading]
    source_file: Optional[str] = None
    file_segment: Optional[int] = None
    rbw_khz: Optional[float] = None

class MeasurementUpdate(BaseModel):
    state: Optional[str] = None
    location: Optional[str] = None
    timestamp: Optional[datetime] = None
    readings: Optional[List[ChannelReading]] = None

class QueryRequest(BaseModel):
    state: str
    location: str
    time: datetime

class QueryResponse(BaseModel):
    channels: List[ChannelReading]
    totalAvailableBandwidth: float
    location: Location
    queryTime: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}

class CSVUploadResponse(BaseModel):
    message: str
    measurements_created: int
    measurements_skipped: int
    location_created: bool
    state_created: bool
    file_name: str
    rows_processed: int
    tvws_rows_processed: int
    segments_found: List[int]


# ---------------------------------------------------------------------------
# Database connection
# ---------------------------------------------------------------------------

client = None
database = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global client, database
    mongodb_url = os.getenv("MONGODB_URL", "mongodb://localhost:27017")
    client = AsyncIOMotorClient(mongodb_url)
    database = client[os.getenv("MONGODB_DB", "tvws_db")]

    await database.users.create_index("email", unique=True)
    await database.states.create_index("name", unique=True)
    await database.locations.create_index([("state", 1), ("name", 1)], unique=True)
    await database.measurements.create_index([("state", 1), ("location", 1), ("timestamp", -1)])
    # Unique index per (location, timestamp, file_segment) to prevent duplicates
    await database.measurements.create_index(
        [("location", 1), ("timestamp", 1), ("file_segment", 1)],
        unique=True,
        sparse=True,
    )

    admin_email = os.getenv("ADMIN_EMAIL", "admin@tvws.ng")
    admin_password = os.getenv("ADMIN_PASSWORD", "admin123")
    admin_user = await database.users.find_one({"email": admin_email})
    if not admin_user:
        password_hash = bcrypt.hashpw(admin_password.encode('utf-8'), bcrypt.gensalt())
        await database.users.insert_one({
            "email": admin_email,
            "password_hash": password_hash.decode('utf-8'),
            "role": "admin",
            "name": "System Administrator",
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow()
        })

    yield
    client.close()


app = FastAPI(
    title="TVWS Geolocation API",
    description="API for managing TV White Space measurements and queries",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# JWT settings
SECRET_KEY = os.getenv("JWT_SECRET_KEY", "your-secret-key-here")
ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "1440"))

security = HTTPBearer()

def create_access_token(data: dict):
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    try:
        token = credentials.credentials
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        if "user_id" not in payload:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token payload")
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Token has expired",
                            headers={"WWW-Authenticate": "Bearer"})
    except jwt.PyJWTError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Could not validate credentials",
                            headers={"WWW-Authenticate": "Bearer"})

async def get_current_user(token_data: dict = Depends(verify_token)):
    user = await database.users.find_one({"_id": ObjectId(token_data["user_id"])})
    if user is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found")
    return user

async def get_admin_user(current_user: dict = Depends(get_current_user)):
    if current_user["role"] != "admin":
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Admin access required")
    return current_user


# ---------------------------------------------------------------------------
# Auth endpoints
# ---------------------------------------------------------------------------

@app.post("/auth/register", response_model=UserResponse)
async def register(user_data: UserCreate):
    existing_user = await database.users.find_one({"email": user_data.email})
    if existing_user:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Email already registered")

    password_hash = bcrypt.hashpw(user_data.password.encode('utf-8'), bcrypt.gensalt())
    user_data_dict = user_data.model_dump(exclude={"password"})
    user_data_dict["password_hash"] = password_hash.decode('utf-8')
    user_data_dict["created_at"] = datetime.utcnow()
    user_data_dict["updated_at"] = datetime.utcnow()

    result = await database.users.insert_one(user_data_dict)
    new_user = await database.users.find_one({"_id": result.inserted_id})
    return UserResponse.from_mongo(new_user)

@app.post("/auth/login")
async def login(user_data: UserLogin):
    user = await database.users.find_one({"email": user_data.email})
    if not user or not bcrypt.checkpw(user_data.password.encode('utf-8'), user["password_hash"].encode('utf-8')):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials",
                            headers={"WWW-Authenticate": "Bearer"})

    access_token_expires = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    token_data = {"user_id": str(user["_id"]), "exp": access_token_expires}
    token = jwt.encode(token_data, SECRET_KEY, algorithm=ALGORITHM)

    return {
        "access_token": token,
        "token_type": "bearer",
        "expires_at": access_token_expires.isoformat(),
        "user": {"id": str(user["_id"]), "email": user["email"], "role": user["role"], "name": user["name"]}
    }

@app.get("/auth/me", response_model=UserResponse)
async def get_me(current_user: dict = Depends(get_current_user)):
    try:
        return UserResponse.from_mongo(current_user)
    except Exception:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED,
                            detail="Could not retrieve user information",
                            headers={"WWW-Authenticate": "Bearer"})


# ---------------------------------------------------------------------------
# Users endpoints
# ---------------------------------------------------------------------------

@app.get("/users", response_model=List[UserResponse])
async def get_users(admin_user: dict = Depends(get_admin_user)):
    return [UserResponse.from_mongo(u) async for u in database.users.find()]

@app.get("/users/{user_id}", response_model=UserResponse)
async def get_user(user_id: str, admin_user: dict = Depends(get_admin_user)):
    if not ObjectId.is_valid(user_id):
        raise HTTPException(status_code=400, detail="Invalid user ID")
    user = await database.users.find_one({"_id": ObjectId(user_id)})
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return UserResponse.from_mongo(user)

@app.put("/users/{user_id}", response_model=UserResponse)
async def update_user(user_id: str, user_data: UserUpdate, admin_user: dict = Depends(get_admin_user)):
    if not ObjectId.is_valid(user_id):
        raise HTTPException(status_code=400, detail="Invalid user ID")
    user = await database.users.find_one({"_id": ObjectId(user_id)})
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    update_data = user_data.model_dump(exclude_unset=True)
    if "password" in update_data:
        update_data["password_hash"] = bcrypt.hashpw(
            update_data.pop("password").encode('utf-8'), bcrypt.gensalt()
        ).decode('utf-8')
    update_data["updated_at"] = datetime.utcnow()

    await database.users.update_one({"_id": ObjectId(user_id)}, {"$set": update_data})
    return UserResponse.from_mongo(await database.users.find_one({"_id": ObjectId(user_id)}))

@app.delete("/users/{user_id}")
async def delete_user(user_id: str, admin_user: dict = Depends(get_admin_user)):
    if not ObjectId.is_valid(user_id):
        raise HTTPException(status_code=400, detail="Invalid user ID")
    result = await database.users.delete_one({"_id": ObjectId(user_id)})
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="User not found")
    return {"message": "User deleted successfully"}


# ---------------------------------------------------------------------------
# States endpoints
# ---------------------------------------------------------------------------

@app.get("/states", response_model=List[State])
async def get_states():
    return [State(**s) async for s in database.states.find()]

@app.post("/states", response_model=State)
async def create_state(state: StateCreate, admin_user: dict = Depends(get_admin_user)):
    if await database.states.find_one({"name": state.name}):
        raise HTTPException(status_code=400, detail="State already exists")
    state_data = {**state.model_dump(), "created_at": datetime.utcnow(), "updated_at": datetime.utcnow()}
    result = await database.states.insert_one(state_data)
    return State.from_mongo(await database.states.find_one({"_id": result.inserted_id}))

@app.get("/states/{state_id}", response_model=State)
async def get_state(state_id: str):
    if not ObjectId.is_valid(state_id):
        raise HTTPException(status_code=400, detail="Invalid state ID")
    state = await database.states.find_one({"_id": ObjectId(state_id)})
    if not state:
        raise HTTPException(status_code=404, detail="State not found")
    return State.from_mongo(state)

@app.put("/states/{state_id}", response_model=State)
async def update_state(state_id: str, state_data: StateUpdate, admin_user: dict = Depends(get_admin_user)):
    if not ObjectId.is_valid(state_id):
        raise HTTPException(status_code=400, detail="Invalid state ID")
    if not await database.states.find_one({"_id": ObjectId(state_id)}):
        raise HTTPException(status_code=404, detail="State not found")
    update_data = {**state_data.model_dump(exclude_unset=True), "updated_at": datetime.utcnow()}
    await database.states.update_one({"_id": ObjectId(state_id)}, {"$set": update_data})
    return State(**await database.states.find_one({"_id": ObjectId(state_id)}))

@app.delete("/states/{state_id}")
async def delete_state(state_id: str, admin_user: dict = Depends(get_admin_user)):
    if not ObjectId.is_valid(state_id):
        raise HTTPException(status_code=400, detail="Invalid state ID")
    if await database.locations.count_documents({"state": state_id}) > 0:
        raise HTTPException(status_code=400, detail="Cannot delete state with associated locations")
    result = await database.states.delete_one({"_id": ObjectId(state_id)})
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="State not found")
    return {"message": "State deleted successfully"}


# ---------------------------------------------------------------------------
# Locations endpoints
# ---------------------------------------------------------------------------

@app.get("/locations", response_model=List[Location])
async def get_locations():
    return [Location(**loc) async for loc in database.locations.find()]

@app.get("/locations/state/{state}", response_model=List[Location])
async def get_locations_by_state(state: str):
    return [Location(**loc) async for loc in database.locations.find({"state": state})]

@app.post("/locations", response_model=Location)
async def create_location(location: LocationCreate, admin_user: dict = Depends(get_admin_user)):
    if await database.locations.find_one({"state": location.state, "name": location.name}):
        raise HTTPException(status_code=400, detail="Location already exists")
    location_data = {**location.model_dump(), "created_at": datetime.utcnow(), "updated_at": datetime.utcnow()}
    result = await database.locations.insert_one(location_data)
    return Location.from_mongo(await database.locations.find_one({"_id": result.inserted_id}))

@app.get("/locations/id/{location_id}", response_model=Location)
async def get_location_by_id(location_id: str):
    if not ObjectId.is_valid(location_id):
        raise HTTPException(status_code=400, detail="Invalid location ID")
    location = await database.locations.find_one({"_id": ObjectId(location_id)})
    if not location:
        raise HTTPException(status_code=404, detail="Location not found")
    return Location.from_mongo(location)

@app.put("/locations/{location_id}", response_model=Location)
async def update_location(location_id: str, location_data: LocationUpdate,
                           admin_user: dict = Depends(get_admin_user)):
    if not ObjectId.is_valid(location_id):
        raise HTTPException(status_code=400, detail="Invalid location ID")
    if not await database.locations.find_one({"_id": ObjectId(location_id)}):
        raise HTTPException(status_code=404, detail="Location not found")
    update_data = {**location_data.model_dump(exclude_unset=True), "updated_at": datetime.utcnow()}
    await database.locations.update_one({"_id": ObjectId(location_id)}, {"$set": update_data})
    return Location(**await database.locations.find_one({"_id": ObjectId(location_id)}))

@app.delete("/locations/{location_id}")
async def delete_location(location_id: str, admin_user: dict = Depends(get_admin_user)):
    if not ObjectId.is_valid(location_id):
        raise HTTPException(status_code=400, detail="Invalid location ID")
    if await database.measurements.count_documents({"location": location_id}) > 0:
        raise HTTPException(status_code=400, detail="Cannot delete location with associated measurements")
    result = await database.locations.delete_one({"_id": ObjectId(location_id)})
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Location not found")
    return {"message": "Location deleted successfully"}


# ---------------------------------------------------------------------------
# CSV Upload - FIXED to accept all frequencies
# ---------------------------------------------------------------------------

# State name lookup by location name prefix (extend as needed).
# If a location name is not found here the upload request must supply it via
# the optional `state_name` query parameter, or it falls back to "Unknown".
LOCATION_STATE_MAP: Dict[str, str] = {
    "Umuahia": "Abia",
    "Aba":     "Abia",
    "Enugu":   "Enugu",
    "Awka":    "Anambra",
    "Owerri":  "Imo",
}

def infer_state(location_name: str) -> str:
    """Best-effort state lookup from the location name prefix."""
    for prefix, state in LOCATION_STATE_MAP.items():
        if location_name.startswith(prefix):
            return state
    return "Unknown"


@app.post("/upload-csv", response_model=CSVUploadResponse)
async def upload_tvws_csv(
    file: UploadFile = File(...),
    state_name: Optional[str] = None,           # caller can supply the state explicitly
    admin_user: dict = Depends(get_admin_user),
):
    """
    Upload a CSV file produced by the RF Explorer spectrum analyser.

    FIX: Accepts ALL frequencies regardless of In_TVWS_Band value.
    Frequencies are automatically mapped to TV channels based on the
    TVWS_CHANNELS definition. Frequencies outside defined channels are skipped.

    Expected columns (from preprocessing pipeline):
        Frequency_MHz, Power_dBm, In_TVWS_Band, File_Segment,
        Location_Name, GPS_Latitude, GPS_Longitude,
        Timestamp_UTC, RBW_kHz, Source_File

    Processing logic
    ----------------
    Each unique (Location_Name, Timestamp_UTC, File_Segment) triplet
    becomes **one Measurement document**. This preserves individual sweeps.

    ALL rows are processed as potential TVWS readings. Each frequency is
    mapped to its parent TV channel (if any). Within each segment the ~0.9 MHz
    spaced readings are binned into their parent 8 MHz TV channel; the
    **maximum power** within that bin is used (worst-case occupancy).
    A channel is marked 'free' when max power is below TVWS_FREE_THRESHOLD_DBM (-97 dBm).
    """
    if not file.filename.lower().endswith('.csv'):
        raise HTTPException(status_code=400, detail="Only CSV files are accepted")

    content = await file.read()
    try:
        text_content = content.decode('utf-8')
    except UnicodeDecodeError:
        text_content = content.decode('latin-1')

    csv_reader = csv.DictReader(io.StringIO(text_content))

    # Validate that the required columns are present
    required_columns = {
        "Frequency_MHz", "Power_dBm", "File_Segment",
        "Location_Name", "GPS_Latitude", "GPS_Longitude",
        "Timestamp_UTC", "RBW_kHz", "Source_File",
    }
    # In_TVWS_Band is optional now - we ignore it
    first_row = None
    rows = []
    for row in csv_reader:
        if first_row is None:
            first_row = row
            missing = required_columns - set(row.keys())
            if missing:
                raise HTTPException(
                    status_code=400,
                    detail=f"CSV is missing required columns: {', '.join(sorted(missing))}"
                )
        rows.append(row)

    if not rows:
        raise HTTPException(status_code=400, detail="CSV file is empty")

    # ------------------------------------------------------------------
    # Pass 1: group raw rows by (location_name, timestamp, file_segment)
    # Process ALL rows - ignore In_TVWS_Band column
    # ------------------------------------------------------------------
    # Structure: segments[location][timestamp_iso][segment_int] = {meta, raw_readings[]}
    segments: Dict[str, Dict[str, Dict[int, Dict]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(lambda: {"meta": None, "raw": []}))
    )

    rows_processed = 0
    tvws_rows_processed = 0
    parse_errors = 0
    frequencies_outside_band = 0

    for row in rows:
        rows_processed += 1
        try:
            freq_mhz    = float(row["Frequency_MHz"])
            power_dbm   = float(row["Power_dBm"])
            segment     = int(row["File_Segment"])
            loc_name    = row["Location_Name"].strip()
            gps_lat     = float(row["GPS_Latitude"])
            gps_lon     = float(row["GPS_Longitude"])
            rbw_khz     = float(row["RBW_kHz"])
            source_file = row["Source_File"].strip()
            timestamp   = datetime.fromisoformat(
                row["Timestamp_UTC"].replace("Z", "+00:00")
            ).replace(tzinfo=None)   # store as naive UTC in MongoDB
        except (ValueError, KeyError) as exc:
            parse_errors += 1
            continue

        ts_iso = timestamp.isoformat()
        bucket = segments[loc_name][ts_iso][segment]

        # Store metadata once per bucket (all rows in a segment share it)
        if bucket["meta"] is None:
            bucket["meta"] = {
                "gps_lat":     gps_lat,
                "gps_lon":     gps_lon,
                "rbw_khz":     rbw_khz,
                "source_file": source_file,
                "timestamp":   timestamp,
                "location":    loc_name,
            }

        # Map frequency to TV channel (skip if outside defined TV bands)
        channel = get_channel_from_frequency(freq_mhz)
        if channel == 0:
            frequencies_outside_band += 1
            continue

        tvws_rows_processed += 1
        bucket["raw"].append({
            "channel":            channel,
            "frequency_mhz":      freq_mhz,
            "signal_strength_dbm": power_dbm,
        })

    # ------------------------------------------------------------------
    # Pass 2: upsert state / location / measurement documents
    # ------------------------------------------------------------------
    measurements_created = 0
    measurements_skipped = 0
    location_created     = False
    state_created        = False
    segments_found: List[int] = []

    for loc_name, ts_map in segments.items():
        resolved_state = state_name or infer_state(loc_name)

        # ---- ensure State exists ----
        if not await database.states.find_one({"name": resolved_state}):
            await database.states.insert_one({
                "name":       resolved_state,
                "created_at": datetime.utcnow(),
                "updated_at": datetime.utcnow(),
            })
            state_created = True

        for ts_iso, seg_map in ts_map.items():
            # Collect GPS from the first segment that has metadata
            any_meta = next(
                (v["meta"] for v in seg_map.values() if v["meta"]), None
            )
            if any_meta is None:
                continue

            # ---- ensure Location exists ----
            existing_loc = await database.locations.find_one(
                {"state": resolved_state, "name": loc_name}
            )
            if not existing_loc:
                loc_result = await database.locations.insert_one({
                    "state": resolved_state,
                    "name":  loc_name,
                    "coordinates": {
                        "lat": any_meta["gps_lat"],
                        "lon": any_meta["gps_lon"],
                    },
                    "created_at": datetime.utcnow(),
                    "updated_at": datetime.utcnow(),
                })
                location_created = True

            # ---- one Measurement per segment ----
            for segment, bucket in seg_map.items():
                if segment not in segments_found:
                    segments_found.append(segment)

                meta = bucket["meta"]
                if meta is None:
                    continue

                # Skip segments with no TVWS readings (all frequencies outside TV band)
                if not bucket["raw"]:
                    continue

                # Aggregate sub-channel readings → one entry per TV channel
                channel_readings = aggregate_readings_to_channels(bucket["raw"])

                # Deduplicate: skip if this (location, timestamp, segment) already exists
                existing_meas = await database.measurements.find_one({
                    "location":     loc_name,
                    "timestamp":    meta["timestamp"],
                    "file_segment": segment,
                })
                if existing_meas:
                    measurements_skipped += 1
                    continue

                await database.measurements.insert_one({
                    "state":        resolved_state,
                    "location":     loc_name,
                    "timestamp":    meta["timestamp"],
                    "readings":     channel_readings,
                    "created_at":   datetime.utcnow(),
                    "source_file":  meta["source_file"],
                    "file_segment": segment,
                    "rbw_khz":      meta["rbw_khz"],
                })
                measurements_created += 1

    # Build message with details about skipped frequencies
    message = f"CSV file processed successfully"
    if parse_errors:
        message += f" ({parse_errors} rows skipped due to parse errors)"
    if frequencies_outside_band:
        message += f" ({frequencies_outside_band} frequencies outside TV band were skipped)"

    return CSVUploadResponse(
        message=message,
        measurements_created=measurements_created,
        measurements_skipped=measurements_skipped,
        location_created=location_created,
        state_created=state_created,
        file_name=file.filename,
        rows_processed=rows_processed,
        tvws_rows_processed=tvws_rows_processed,
        segments_found=sorted(segments_found),
    )


# ---------------------------------------------------------------------------
# Measurements endpoints
# ---------------------------------------------------------------------------

@app.get("/measurements", response_model=List[Measurement])
async def get_measurements(admin_user: dict = Depends(get_admin_user)):
    return [Measurement(**m) async for m in database.measurements.find()]

@app.post("/measurements", response_model=Measurement)
async def upload_measurements(measurement: MeasurementCreate, admin_user: dict = Depends(get_admin_user)):
    processed_readings = [
        {
            "channel":            r.channel,
            "frequency_mhz":      r.frequency_mhz,
            "signal_strength_dbm": r.signal_strength_dbm,
            "status":             "free" if r.signal_strength_dbm < TVWS_FREE_THRESHOLD_DBM else "occupied",
        }
        for r in measurement.readings
    ]

    measurement_data = {
        "state":        measurement.state,
        "location":     measurement.location,
        "timestamp":    measurement.timestamp,
        "readings":     processed_readings,
        "created_at":   datetime.utcnow(),
        "source_file":  measurement.source_file,
        "file_segment": measurement.file_segment,
        "rbw_khz":      measurement.rbw_khz,
    }
    result = await database.measurements.insert_one(measurement_data)
    return Measurement.from_mongo(await database.measurements.find_one({"_id": result.inserted_id}))

@app.get("/measurements/{measurement_id}", response_model=Measurement)
async def get_measurement(measurement_id: str, admin_user: dict = Depends(get_admin_user)):
    if not ObjectId.is_valid(measurement_id):
        raise HTTPException(status_code=400, detail="Invalid measurement ID")
    measurement = await database.measurements.find_one({"_id": ObjectId(measurement_id)})
    if not measurement:
        raise HTTPException(status_code=404, detail="Measurement not found")
    return Measurement.from_mongo(measurement)


# Additional Measurement GET endpoints

@app.get("/measurements/all", response_model=List[Measurement])
async def get_all_measurements(
    skip: int = 0,
    limit: int = 100,
    admin_user: dict = Depends(get_admin_user)
):
    """Get all measurements with pagination"""
    measurements = []
    cursor = database.measurements.find().sort("timestamp", -1).skip(skip).limit(limit)
    async for m in cursor:
        measurements.append(Measurement.from_mongo(m))
    return measurements


@app.get("/measurements/by-location/{location_name}", response_model=List[Measurement])
async def get_measurements_by_location_name(
    location_name: str,
    state: Optional[str] = None,
    limit: int = 50,
    admin_user: dict = Depends(get_admin_user)
):
    """Get measurements by location name, optionally filtered by state"""
    query = {"location": location_name}
    if state:
        query["state"] = state
    measurements = []
    async for m in database.measurements.find(query).sort("timestamp", -1).limit(limit):
        measurements.append(Measurement.from_mongo(m))
    return measurements


@app.get("/measurements/by-state/{state_name}", response_model=List[Measurement])
async def get_measurements_by_state(
    state_name: str,
    limit: int = 100,
    admin_user: dict = Depends(get_admin_user)
):
    """Get all measurements for a specific state"""
    measurements = []
    async for m in database.measurements.find({"state": state_name}).sort("timestamp", -1).limit(limit):
        measurements.append(Measurement.from_mongo(m))
    return measurements


@app.get("/measurements/by-location-id/{location_id}", response_model=List[Measurement])
async def get_measurements_by_location_id(
    location_id: str,
    limit: int = 50,
    admin_user: dict = Depends(get_admin_user)
):
    """Get measurements by location ID"""
    if not ObjectId.is_valid(location_id):
        raise HTTPException(status_code=400, detail="Invalid location ID")
    location = await database.locations.find_one({"_id": ObjectId(location_id)})
    if not location:
        raise HTTPException(status_code=404, detail="Location not found")
    measurements = []
    async for m in database.measurements.find(
        {"location": location["name"], "state": location["state"]}
    ).sort("timestamp", -1).limit(limit):
        measurements.append(Measurement.from_mongo(m))
    return measurements


@app.get("/measurements/latest/{location_name}", response_model=Optional[Measurement])
async def get_latest_measurement(location_name: str, state: Optional[str] = None):
    """
    Get the latest merged TVWS measurement for a location (public endpoint).

    When a location has multiple segments, this endpoint returns a synthetic
    'merged' measurement that combines the channel readings from all segments
    at the most recent timestamp, giving a full picture of spectrum occupancy.
    """
    query = {"location": location_name}
    if state:
        query["state"] = state

    # Find the latest timestamp for this location
    latest = await database.measurements.find_one(query, sort=[("timestamp", -1)])
    if not latest:
        raise HTTPException(status_code=404, detail="No measurements found for this location")

    latest_ts = latest["timestamp"]

    # Collect all segments at that timestamp
    all_readings: Dict[int, Dict] = {}
    async for m in database.measurements.find(
        {**query, "timestamp": latest_ts}
    ):
        for reading in m["readings"]:
            ch = reading["channel"]
            # If channel appears in multiple segments, keep the worst-case (highest power)
            if ch not in all_readings or reading["signal_strength_dbm"] > all_readings[ch]["signal_strength_dbm"]:
                all_readings[ch] = reading

    merged = latest.copy()
    merged["readings"] = sorted(all_readings.values(), key=lambda r: r["channel"])
    merged["file_segment"] = None   # merged across segments

    return Measurement.from_mongo(merged)


@app.get("/measurements/summary/{state_name}", response_model=Dict)
async def get_state_summary(state_name: str, admin_user: dict = Depends(get_admin_user)):
    """Get summary statistics for a state"""
    total_measurements   = 0
    unique_locations     = set()
    total_free_channels  = 0
    total_occupied       = 0
    channel_stats        = defaultdict(lambda: {"free": 0, "occupied": 0, "total": 0})

    async for m in database.measurements.find({"state": state_name}):
        total_measurements += 1
        unique_locations.add(m["location"])
        for reading in m["readings"]:
            ch     = reading.get("channel", 0)
            status = reading.get("status", "unknown")
            channel_stats[ch]["total"] += 1
            if status == "free":
                total_free_channels += 1
                channel_stats[ch]["free"] += 1
            elif status == "occupied":
                total_occupied += 1
                channel_stats[ch]["occupied"] += 1

    locations = []
    async for loc in database.locations.find({"state": state_name}):
        locations.append({
            "id":          str(loc["_id"]),
            "name":        loc["name"],
            "coordinates": loc["coordinates"],
        })

    return {
        "state":              state_name,
        "total_measurements": total_measurements,
        "unique_locations":   len(unique_locations),
        "locations":          locations,
        "channel_statistics": {str(ch): stats for ch, stats in channel_stats.items()},
        "summary": {
            "total_free_channel_readings":     total_free_channels,
            "total_occupied_channel_readings": total_occupied,
            "average_free_per_measurement":    (
                total_free_channels / total_measurements if total_measurements else 0
            ),
        },
    }


@app.get("/measurements/date-range", response_model=List[Measurement])
async def get_measurements_by_date_range(
    start_date: str,
    end_date: str,
    state: Optional[str] = None,
    location: Optional[str] = None,
    admin_user: dict = Depends(get_admin_user)
):
    """Get measurements within a date range"""
    try:
        start = datetime.fromisoformat(start_date.replace('Z', '+00:00')).replace(tzinfo=None)
        end   = datetime.fromisoformat(end_date.replace('Z', '+00:00')).replace(tzinfo=None)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid date format. Use ISO format (YYYY-MM-DDTHH:MM:SSZ)")

    query: Dict = {"timestamp": {"$gte": start, "$lte": end}}
    if state:
        query["state"] = state
    if location:
        query["location"] = location

    measurements = []
    async for m in database.measurements.find(query).sort("timestamp", -1):
        measurements.append(Measurement.from_mongo(m))
    return measurements


@app.get("/measurements/stats/free-channels", response_model=Dict)
async def get_free_channels_stats(
    state: Optional[str] = None,
    location: Optional[str] = None,
    admin_user: dict = Depends(get_admin_user)
):
    """Get statistics about free channels across measurements"""
    query: Dict = {}
    if state:
        query["state"] = state
    if location:
        query["location"] = location

    pipeline = [
        {"$match": query},
        {"$unwind": "$readings"},
        {"$match": {"readings.status": "free"}},
        {"$group": {
            "_id":        {"channel": "$readings.channel", "frequency": "$readings.frequency_mhz"},
            "count":      {"$sum": 1},
            "avg_signal": {"$avg": "$readings.signal_strength_dbm"},
        }},
        {"$sort": {"_id.channel": 1}},
    ]

    results = []
    async for doc in database.measurements.aggregate(pipeline):
        results.append({
            "channel":          doc["_id"]["channel"],
            "frequency_mhz":    doc["_id"]["frequency"],
            "times_free":       doc["count"],
            "average_signal_dbm": round(doc["avg_signal"], 2),
        })

    return {
        "total_free_channel_occurrences": sum(r["times_free"] for r in results),
        "unique_free_channels":           len(results),
        "channels":                       results,
    }


@app.get("/measurements/export/{location_name}", response_model=Dict)
async def export_measurements(
    location_name: str,
    state: Optional[str] = None,
    format: str = "json",
    admin_user: dict = Depends(get_admin_user)
):
    """Export measurements for a location in JSON or CSV format"""
    query = {"location": location_name}
    if state:
        query["state"] = state

    measurements = []
    async for m in database.measurements.find(query).sort("timestamp", -1):
        measurements.append({
            "id":           str(m["_id"]),
            "state":        m["state"],
            "location":     m["location"],
            "timestamp":    m["timestamp"].isoformat(),
            "file_segment": m.get("file_segment"),
            "rbw_khz":      m.get("rbw_khz"),
            "readings":     m["readings"],
            "created_at":   m.get("created_at", m["timestamp"]).isoformat(),
        })

    if format == "csv":
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow([
            "Measurement ID", "State", "Location", "Timestamp",
            "File_Segment", "RBW_kHz",
            "Channel", "Frequency_MHz", "Signal_dBm", "Status",
        ])
        for m in measurements:
            for reading in m["readings"]:
                writer.writerow([
                    m["id"], m["state"], m["location"], m["timestamp"],
                    m["file_segment"], m["rbw_khz"],
                    reading["channel"], reading["frequency_mhz"],
                    reading["signal_strength_dbm"], reading["status"],
                ])
        return {"csv_data": output.getvalue()}

    return {
        "location":          location_name,
        "state":             state or "all",
        "total_measurements": len(measurements),
        "measurements":      measurements,
    }


@app.put("/measurements/{measurement_id}", response_model=Measurement)
async def update_measurement(
    measurement_id: str,
    measurement_data: MeasurementUpdate,
    admin_user: dict = Depends(get_admin_user)
):
    if not ObjectId.is_valid(measurement_id):
        raise HTTPException(status_code=400, detail="Invalid measurement ID")
    if not await database.measurements.find_one({"_id": ObjectId(measurement_id)}):
        raise HTTPException(status_code=404, detail="Measurement not found")

    update_data = measurement_data.model_dump(exclude_unset=True)
    if "readings" in update_data:
        update_data["readings"] = [
            {
                "channel":            r["channel"],
                "frequency_mhz":      r["frequency_mhz"],
                "signal_strength_dbm": r["signal_strength_dbm"],
                "status":             "free" if r["signal_strength_dbm"] < TVWS_FREE_THRESHOLD_DBM else "occupied",
            }
            for r in update_data["readings"]
        ]

    await database.measurements.update_one({"_id": ObjectId(measurement_id)}, {"$set": update_data})
    return Measurement(**await database.measurements.find_one({"_id": ObjectId(measurement_id)}))


@app.delete("/measurements/{measurement_id}")
async def delete_measurement(measurement_id: str, admin_user: dict = Depends(get_admin_user)):
    if not ObjectId.is_valid(measurement_id):
        raise HTTPException(status_code=400, detail="Invalid measurement ID")
    result = await database.measurements.delete_one({"_id": ObjectId(measurement_id)})
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Measurement not found")
    return {"message": "Measurement deleted successfully"}


# ---------------------------------------------------------------------------
# TVWS Query endpoint
# ---------------------------------------------------------------------------

@app.post("/query-tvws", response_model=QueryResponse)
async def query_tvws(query: QueryRequest):
    """
    Query TVWS channel availability at a specific location and time.

    Returns a merged view across all segments recorded at or before the
    requested time, giving a complete channel availability picture.
    """
    location = await database.locations.find_one({"state": query.state, "name": query.location})
    if not location:
        raise HTTPException(status_code=404, detail="Location not found")

    # Find the latest timestamp at or before query time
    latest = await database.measurements.find_one(
        {"state": query.state, "location": query.location, "timestamp": {"$lte": query.time}},
        sort=[("timestamp", -1)]
    )

    if not latest:
        # No data — return all channels as unknown
        channels = [
            ChannelReading(
                channel=ch,
                frequency_mhz=float(freq),
                signal_strength_dbm=0.0,
                status="unknown"
            )
            for ch, freq in TVWS_CHANNELS.items()
        ]
        return QueryResponse(
            channels=channels,
            totalAvailableBandwidth=0.0,
            location=Location(**location),
            queryTime=query.time
        )

    latest_ts = latest["timestamp"]

    # Merge all segments at that timestamp (worst-case per channel)
    merged: Dict[int, Dict] = {}
    async for m in database.measurements.find(
        {"state": query.state, "location": query.location, "timestamp": latest_ts}
    ):
        for reading in m["readings"]:
            ch = reading["channel"]
            if ch not in merged or reading["signal_strength_dbm"] > merged[ch]["signal_strength_dbm"]:
                merged[ch] = reading

    channel_list = sorted(merged.values(), key=lambda r: r["channel"])
    free_count   = sum(1 for r in channel_list if r.get("status") == "free")

    return QueryResponse(
        channels=channel_list,
        totalAvailableBandwidth=float(free_count * 8),   # 8 MHz per channel
        location=Location(**location),
        queryTime=query.time
    )


# ---------------------------------------------------------------------------
# Utility endpoints
# ---------------------------------------------------------------------------

@app.get("/tvws-channels")
async def get_tvws_channels():
    """List all defined TVWS channels with their center frequencies."""
    return {
        "channels":      [{"channel": ch, "frequency_mhz": freq} for ch, freq in TVWS_CHANNELS.items()],
        "bandwidth_mhz": 8,
        "total_channels": len(TVWS_CHANNELS),
    }


@app.get("/measurements/location/{state}/{location}")
async def get_measurements_by_location(
    state: str,
    location: str,
    limit: int = 10,
    admin_user: dict = Depends(get_admin_user)
):
    """Get measurements for a specific state/location combination."""
    measurements = []
    async for m in database.measurements.find(
        {"state": state, "location": location}
    ).sort("timestamp", -1).limit(limit):
        measurements.append(Measurement.from_mongo(m))
    return measurements


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
