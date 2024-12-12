from typing import Optional
from motor.motor_asyncio import AsyncIOMotorClient
from pymongo.errors import ConnectionFailure
import asyncio
from ..utils.logging import kpi_logger

class Database:
    def __init__(self, uri: str, database_name: str):
        self.client: Optional[AsyncIOMotorClient] = None
        self.uri = uri
        self.database_name = database_name
        self._connection_lock = asyncio.Lock()
        self._connected = False

    async def connect(self) -> None:
        """Connect to database"""
        if self._connected:
            return

        async with self._connection_lock:
            if self._connected:
                return

            try:
                self.client = AsyncIOMotorClient(self.uri)
                # Verify connection
                await self.client.admin.command('ping')
                self._connected = True
                kpi_logger.info("Successfully connected to database")
            except ConnectionFailure as e:
                kpi_logger.error(f"Failed to connect to database: {str(e)}")
                raise

    async def disconnect(self) -> None:
        """Disconnect from database"""
        if self.client:
            self.client.close()
            self._connected = False
            kpi_logger.info("Disconnected from database")

    @property
    def db(self):
        """Get database instance"""
        if not self._connected:
            raise ConnectionError("Not connected to database")
        return self.client[self.database_name]

    async def ensure_indexes(self) -> None:
        """Create required indexes"""
        # KPI indexes
        await self.db.kpis.create_index("id", unique=True)
        await self.db.kpis.create_index("metadata.created_by")
        await self.db.kpis.create_index("metadata.modified_at")
        await self.db.kpis.create_index("status")
        await self.db.kpis.create_index("type")
        
        # Assignment indexes
        await self.db.kpis.create_index("assignment.team_assignments")
        await self.db.kpis.create_index("assignment.project_assignments")
        await self.db.kpis.create_index("assignment.user_assignments")
        await self.db.kpis.create_index("assignment.department_assignments")
        await self.db.kpis.create_index("assignment.business_unit_assignments")

class DatabaseSession:
    """Context manager for database operations"""
    def __init__(self, database: Database):
        self.database = database

    async def __aenter__(self):
        await self.database.connect()
        return self.database.db

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if exc_type:
            kpi_logger.error(
                f"Error during database operation: {str(exc_val)}"
            )
        # Don't disconnect as other operations might be ongoing
        return False

async def get_database():
    """Database dependency for FastAPI"""
    database = Database(
        uri="mongodb://localhost:27017",  # Configure from environment
        database_name="kpi_management"    # Configure from environment
    )
    try:
        await database.connect()
        yield database
    finally:
        await database.disconnect()
