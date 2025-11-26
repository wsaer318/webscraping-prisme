from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, ForeignKey
from sqlalchemy.orm import declarative_base, sessionmaker, relationship
from datetime import datetime
import os

# Configuration (identique à src/database.py pour la migration)
DB_PATH = os.path.join("data", "prisma.db")
DATABASE_URL = f"sqlite:///{DB_PATH}"

Base = declarative_base()

class ArticleHistory(Base):
    """Historique des changements d'état d'un article pour traçabilité PRISMA"""
    __tablename__ = "article_history"
    
    id = Column(Integer, primary_key=True, index=True)
    article_id = Column(Integer, ForeignKey("articles.id"), nullable=False)
    
    previous_status = Column(String)
    new_status = Column(String, nullable=False)
    
    action = Column(String) # "SCREENING_INCLUDE", "SCREENING_EXCLUDE", "ELIGIBILITY_REJECT", etc.
    reason = Column(String) # Raison de l'exclusion ou note
    
    timestamp = Column(DateTime, default=datetime.utcnow)
    user = Column(String, default="System") # Pour futur multi-user

def migrate():
    print(f"Migration de la base de données : {DB_PATH}")
    engine = create_engine(DATABASE_URL)
    
    # Créer la table si elle n'existe pas
    Base.metadata.create_all(engine)
    print("✅ Table 'article_history' créée avec succès.")

if __name__ == "__main__":
    migrate()
