from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
from pydantic import BaseModel, Field, ConfigDict, EmailStr
from typing import List, Optional, Annotated, Any
from datetime import datetime, timezone, timedelta
from bson import ObjectId
import os
from jose import JWTError, jwt
import bcrypt
from contextlib import asynccontextmanager
from dotenv import load_dotenv
from pydantic.functional_validators import BeforeValidator
from typing_extensions import TypedDict

# Load environment variables
load_dotenv()


# Pydantic models
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


class MeasurementCreate(BaseModel):
    state: str
    location: str
    timestamp: datetime
    readings: List[ChannelReading]


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
    queryTime: str


# Database connection
client = None
database = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    global client, database
    mongodb_url = os.getenv("MONGODB_URL", "mongodb://localhost:27017")
    client = AsyncIOMotorClient(mongodb_url)
    database = client[os.getenv("MONGODB_DB", "tvws_db")]

    # Create indexes
    await database.users.create_index("email", unique=True)
    await database.states.create_index("name", unique=True)
    await database.locations.create_index([("state", 1), ("name", 1)], unique=True)
    await database.measurements.create_index([("state", 1), ("location", 1), ("timestamp", -1)])

    # Create default admin user if not exists
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

    # Shutdown
    client.close()


app = FastAPI(
    title="TVWS Geolocation API",
    description="API for managing TV White Space measurements and queries",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
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
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "1440"))  # 24 hours

security = HTTPBearer()


def create_access_token(data: dict):
    to_encode = data.copy()
    expire = datetime.utcnow() + timezone.utc.utcoffset(datetime.utcnow()) + timezone.utc.utcoffset(
        datetime.utcnow()) * ACCESS_TOKEN_EXPIRE_MINUTES
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    try:
        token = credentials.credentials
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        if "user_id" not in payload:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token payload"
            )
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has expired",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except jwt.PyJWTError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )

async def get_current_user(token_data: dict = Depends(verify_token)):
    user = await database.users.find_one({"_id": ObjectId(token_data["user_id"])})
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found"
        )
    return user


async def get_admin_user(current_user: dict = Depends(get_current_user)):
    if current_user["role"] != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required"
        )
    return current_user


# Authentication endpoints
@app.post("/auth/register", response_model=UserResponse)
async def register(user_data: UserCreate):
    existing_user = await database.users.find_one({"email": user_data.email})
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered"
        )

    password_hash = bcrypt.hashpw(user_data.password.encode('utf-8'), bcrypt.gensalt())
    user_data_dict = user_data.dict(exclude={"password"})
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
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Set token expiration
    access_token_expires = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    token_data = {
        "user_id": str(user["_id"]),
        "exp": access_token_expires
    }
    token = jwt.encode(token_data, SECRET_KEY, algorithm=ALGORITHM)

    return {
        "access_token": token,
        "token_type": "bearer",
        "expires_at": access_token_expires.isoformat(),
        "user": {
            "id": str(user["_id"]),
            "email": user["email"],
            "role": user["role"],
            "name": user["name"]
        }
    }


@app.get("/auth/me", response_model=UserResponse)
async def get_me(current_user: dict = Depends(get_current_user)):
    try:
        return UserResponse.from_mongo(current_user)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not retrieve user information",
            headers={"WWW-Authenticate": "Bearer"},
        )

# Users endpoints
@app.get("/users", response_model=List[UserResponse])
async def get_users(admin_user: dict = Depends(get_admin_user)):
    users = []
    async for user in database.users.find():
        users.append(UserResponse.from_mongo(user))
    return users

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

    update_data = user_data.dict(exclude_unset=True)
    if "password" in update_data:
        password_hash = bcrypt.hashpw(update_data["password"].encode('utf-8'), bcrypt.gensalt())
        update_data["password_hash"] = password_hash.decode('utf-8')
        del update_data["password"]

    update_data["updated_at"] = datetime.utcnow()

    await database.users.update_one(
        {"_id": ObjectId(user_id)},
        {"$set": update_data}
    )
    updated_user = await database.users.find_one({"_id": ObjectId(user_id)})
    return UserResponse.from_mongo(updated_user)


@app.delete("/users/{user_id}")
async def delete_user(user_id: str, admin_user: dict = Depends(get_admin_user)):
    if not ObjectId.is_valid(user_id):
        raise HTTPException(status_code=400, detail="Invalid user ID")

    result = await database.users.delete_one({"_id": ObjectId(user_id)})
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="User not found")
    return {"message": "User deleted successfully"}



# States endpoints
@app.get("/states", response_model=List[State])
async def get_states():
    states = []
    async for state in database.states.find():
        states.append(State(**state))
    return states


@app.post("/states", response_model=State)
async def create_state(state: StateCreate, admin_user: dict = Depends(get_admin_user)):
    existing = await database.states.find_one({"name": state.name})
    if existing:
        raise HTTPException(status_code=400, detail="State already exists")

    state_data = state.model_dump()
    state_data["created_at"] = datetime.utcnow()
    state_data["updated_at"] = datetime.utcnow()

    result = await database.states.insert_one(state_data)
    new_state = await database.states.find_one({"_id": result.inserted_id})
    return State.from_mongo(new_state)


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

    state = await database.states.find_one({"_id": ObjectId(state_id)})
    if not state:
        raise HTTPException(status_code=404, detail="State not found")

    update_data = state_data.dict(exclude_unset=True)
    update_data["updated_at"] = datetime.utcnow()

    await database.states.update_one(
        {"_id": ObjectId(state_id)},
        {"$set": update_data}
    )
    updated_state = await database.states.find_one({"_id": ObjectId(state_id)})
    return State(**updated_state)


@app.delete("/states/{state_id}")
async def delete_state(state_id: str, admin_user: dict = Depends(get_admin_user)):
    if not ObjectId.is_valid(state_id):
        raise HTTPException(status_code=400, detail="Invalid state ID")

    # Check if there are locations associated with this state
    locations_count = await database.locations.count_documents({"state": state_id})
    if locations_count > 0:
        raise HTTPException(
            status_code=400,
            detail="Cannot delete state with associated locations"
        )

    result = await database.states.delete_one({"_id": ObjectId(state_id)})
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="State not found")
    return {"message": "State deleted successfully"}


# Locations endpoints
@app.get("/locations", response_model=List[Location])
async def get_locations():
    locations = []
    async for location in database.locations.find():
        locations.append(Location(**location))
    return locations


@app.get("/locations/{state}", response_model=List[Location])
async def get_locations_by_state(state: str):
    locations = []
    async for location in database.locations.find({"state": state}):
        locations.append(Location(**location))
    return locations


@app.post("/locations", response_model=Location)
async def create_location(location: LocationCreate, admin_user: dict = Depends(get_admin_user)):
    existing = await database.locations.find_one({
        "state": location.state,
        "name": location.name
    })
    if existing:
        raise HTTPException(status_code=400, detail="Location already exists")

    location_data = location.model_dump()
    location_data["created_at"] = datetime.utcnow()
    location_data["updated_at"] = datetime.utcnow()

    result = await database.locations.insert_one(location_data)
    new_location = await database.locations.find_one({"_id": result.inserted_id})
    return Location.from_mongo(new_location)


@app.get("/locations/id/{location_id}", response_model=Location)
async def get_location_by_id(location_id: str):
    if not ObjectId.is_valid(location_id):
        raise HTTPException(status_code=400, detail="Invalid location ID")

    location = await database.locations.find_one({"_id": ObjectId(location_id)})
    if not location:
        raise HTTPException(status_code=404, detail="Location not found")
    return Location.from_mongo(location)


@app.put("/locations/{location_id}", response_model=Location)
async def update_location(location_id: str, location_data: LocationUpdate, admin_user: dict = Depends(get_admin_user)):
    if not ObjectId.is_valid(location_id):
        raise HTTPException(status_code=400, detail="Invalid location ID")

    location = await database.locations.find_one({"_id": ObjectId(location_id)})
    if not location:
        raise HTTPException(status_code=404, detail="Location not found")

    update_data = location_data.dict(exclude_unset=True)
    update_data["updated_at"] = datetime.utcnow()

    await database.locations.update_one(
        {"_id": ObjectId(location_id)},
        {"$set": update_data}
    )
    updated_location = await database.locations.find_one({"_id": ObjectId(location_id)})
    return Location(**updated_location)


@app.delete("/locations/{location_id}")
async def delete_location(location_id: str, admin_user: dict = Depends(get_admin_user)):
    if not ObjectId.is_valid(location_id):
        raise HTTPException(status_code=400, detail="Invalid location ID")

    # Check if there are measurements associated with this location
    measurements_count = await database.measurements.count_documents({"location": location_id})
    if measurements_count > 0:
        raise HTTPException(
            status_code=400,
            detail="Cannot delete location with associated measurements"
        )

    result = await database.locations.delete_one({"_id": ObjectId(location_id)})
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Location not found")
    return {"message": "Location deleted successfully"}


# Measurements endpoints
@app.get("/measurements", response_model=List[Measurement])
async def get_measurements(admin_user: dict = Depends(get_admin_user)):
    measurements = []
    async for measurement in database.measurements.find():
        measurements.append(Measurement(**measurement))
    return measurements


@app.post("/measurements", response_model=Measurement)
async def upload_measurements(measurement: MeasurementCreate, admin_user: dict = Depends(get_admin_user)):
    processed_readings = []
    for reading in measurement.readings:
        status = "free" if reading.signal_strength_dbm < -97 else "occupied"
        processed_readings.append({
            "channel": reading.channel,
            "frequency_mhz": reading.frequency_mhz,
            "signal_strength_dbm": reading.signal_strength_dbm,
            "status": status
        })

    measurement_data = {
        "state": measurement.state,
        "location": measurement.location,
        "timestamp": measurement.timestamp,
        "readings": processed_readings,
        "created_at": datetime.utcnow()
    }

    result = await database.measurements.insert_one(measurement_data)
    new_measurement = await database.measurements.find_one({"_id": result.inserted_id})
    return Measurement.from_mongo(new_measurement)


@app.get("/measurements/{measurement_id}", response_model=Measurement)
async def get_measurement(measurement_id: str, admin_user: dict = Depends(get_admin_user)):
    if not ObjectId.is_valid(measurement_id):
        raise HTTPException(status_code=400, detail="Invalid measurement ID")

    measurement = await database.measurements.find_one({"_id": ObjectId(measurement_id)})
    if not measurement:
        raise HTTPException(status_code=404, detail="Measurement not found")
    return Measurement.from_mongo(measurement)


@app.put("/measurements/{measurement_id}", response_model=Measurement)
async def update_measurement(
        measurement_id: str,
        measurement_data: MeasurementUpdate,
        admin_user: dict = Depends(get_admin_user)
):
    if not ObjectId.is_valid(measurement_id):
        raise HTTPException(status_code=400, detail="Invalid measurement ID")

    measurement = await database.measurements.find_one({"_id": ObjectId(measurement_id)})
    if not measurement:
        raise HTTPException(status_code=404, detail="Measurement not found")

    update_data = measurement_data.dict(exclude_unset=True)

    # Process readings if they are being updated
    if "readings" in update_data:
        processed_readings = []
        for reading in update_data["readings"]:
            status = "free" if reading["signal_strength_dbm"] < -97 else "occupied"
            processed_readings.append({
                "channel": reading["channel"],
                "frequency_mhz": reading["frequency_mhz"],
                "signal_strength_dbm": reading["signal_strength_dbm"],
                "status": status
            })
        update_data["readings"] = processed_readings

    await database.measurements.update_one(
        {"_id": ObjectId(measurement_id)},
        {"$set": update_data}
    )
    updated_measurement = await database.measurements.find_one({"_id": ObjectId(measurement_id)})
    return Measurement(**updated_measurement)


@app.delete("/measurements/{measurement_id}")
async def delete_measurement(measurement_id: str, admin_user: dict = Depends(get_admin_user)):
    if not ObjectId.is_valid(measurement_id):
        raise HTTPException(status_code=400, detail="Invalid measurement ID")

    result = await database.measurements.delete_one({"_id": ObjectId(measurement_id)})
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Measurement not found")
    return {"message": "Measurement deleted successfully"}


@app.post("/query-tvws", response_model=QueryResponse)
async def query_tvws(query: QueryRequest):
    # Find location
    location = await database.locations.find_one({
        "state": query.state,
        "name": query.location
    })

    if not location:
        raise HTTPException(status_code=404, detail="Location not found")

    # Find the most recent measurement for this location
    measurement = await database.measurements.find_one(
        {
            "state": query.state,
            "location": query.location,
            "timestamp": {"$lte": query.time}
        },
        sort=[("timestamp", -1)]
    )

    if not measurement:
        raise HTTPException(
            status_code=404,
            detail="No measurements found for this location and time"
        )

    # Calculate total available bandwidth
    free_channels = [ch for ch in measurement["readings"] if ch["status"] == "free"]
    total_bandwidth = len(free_channels) * 8  # 8 MHz per channel

    return QueryResponse(
        channels=measurement["readings"],
        totalAvailableBandwidth=total_bandwidth,
        location=Location(**location),
        queryTime=query.time.isoformat()
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8000")))