import os
from sqlalchemy import create_engine, Column, Integer, String, Text, Boolean, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker
from datetime import datetime

# Configuration
DB_PATH = os.path.join("data", "prisma.db")
os.makedirs("data", exist_ok=True)
DATABASE_URL = f"sqlite:///{DB_PATH}"

Base = declarative_base()

class Article(Base):
    __tablename__ = "articles"

    id = Column(Integer, primary_key=True, index=True)
    
    # Metadata
    source = Column(String, default="Google Scholar")
    title = Column(String, nullable=False)
    authors = Column(String)
    year = Column(Integer)
    link = Column(String)
    abstract = Column(Text)
    
    # Files
    pdf_path = Column(String, nullable=True)
    
    # PRISMA Status
    # Statuses: 'IDENTIFIED', 'SCREENED_IN', 'EXCLUDED_SCREENING', 'ELIGIBLE', 'EXCLUDED_ELIGIBILITY', 'INCLUDED'
    status = Column(String, default="IDENTIFIED", index=True)
    
    # Decisions
    exclusion_reason = Column(String, nullable=True)
    notes = Column(Text, nullable=True)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def __repr__(self):
        return f"<Article(id={self.id}, title='{self.title[:30]}...', status='{self.status}')>"

# Database Setup
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def init_db():
    Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
