import os
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, ForeignKey, Float
from sqlalchemy.orm import declarative_base, sessionmaker, relationship
from datetime import datetime

# Configuration
DB_PATH = os.path.join("data", "prisma.db")
os.makedirs("data", exist_ok=True)
DATABASE_URL = f"sqlite:///{DB_PATH}"

Base = declarative_base()


class SearchSession(Base):
    """Session de recherche pour grouper les articles d'une même requête"""
    __tablename__ = 'search_sessions'
    
    id = Column(Integer, primary_key=True)
    query = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    num_results = Column(Integer, default=0)
    successful_downloads = Column(Integer, default=0)
    status = Column(String, default='ACTIVE')  # ACTIVE, ARCHIVED, DELETED
    
    # Relation inverse
    articles = relationship("Article", back_populates="session")


class Article(Base):
    __tablename__ = "articles"

    id = Column(Integer, primary_key=True, index=True)
    
    # Metadata
    source = Column(String, default="Google Scholar")
    title = Column(String, nullable=False)
    authors = Column(String)
    year = Column(Integer)
    link = Column(String)
    doi = Column(String, nullable=True, index=True)  # Digital Object Identifier (unique)
    abstract = Column(Text)
    
    # Files
    pdf_path = Column(String, nullable=True)
    
    # Text extraction
    full_text = Column(Text, nullable=True)
    text_extraction_status =Column(String, default="NOT_ATTEMPTED")
    extraction_method = Column(String, nullable=True)
    
    # AI Analysis
    relevance_score = Column(Float, nullable=True)
    suggested_reason = Column(String, nullable=True) # Raison suggérée par le Cross-Encoder
    ia_metadata = Column(Text, nullable=True) # JSON string pour détails (méthode, confiance, chunks)
    
    # PRISMA Status
    status = Column(String, default="IDENTIFIED", index=True)
    
    # Screening Decisions
    exclusion_reason = Column(String, nullable=True)
    notes = Column(Text, nullable=True)
    
    # Eligibility Phase (Phase 3)
    eligibility_notes = Column(Text, nullable=True)
    reviewed_at = Column(DateTime, nullable=True)
    reviewer = Column(String, nullable=True)
    
    # Additional PDF metadata
    pdf_url = Column(String, nullable=True)
    arxiv_id = Column(String, nullable=True)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Session de recherche
    search_session_id = Column(Integer, ForeignKey('search_sessions.id'), nullable=True)
    session = relationship("SearchSession", back_populates="articles")
    
    # Historique
    history = relationship("ArticleHistory", back_populates="article", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<Article(id={self.id}, title='{self.title[:30]}...', status='{self.status}')>"


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
    user = Column(String, default="System")
    
    # Relation
    article = relationship("Article", back_populates="history")


class ExclusionCriteria(Base):
    """Critères d'exclusion pour l'IA (Cross-Encoder) - Phase Screening"""
    __tablename__ = "exclusion_criteria"
    
    id = Column(Integer, primary_key=True, index=True)
    label = Column(String, nullable=False) # Ex: "Animal Study"
    description = Column(String, nullable=False) # Ex: "Study performed on animals..."
    active = Column(Integer, default=1) # 1=Active, 0=Inactive (Integer for SQLite boolean)
    
    created_at = Column(DateTime, default=datetime.utcnow)


class EligibilityCriteria(Base):
    """Critères d'éligibilité pour Phase 3 - Revue texte complet"""
    __tablename__ = "eligibility_criteria"
    
    id = Column(Integer, primary_key=True, index=True)
    label = Column(String, nullable=False)
    description = Column(Text, nullable=False)
    type = Column(String, nullable=False) # "INCLUSION" ou "EXCLUSION"
    active = Column(Integer, default=1)
    
    created_at = Column(DateTime, default=datetime.utcnow)


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


def delete_article_with_cleanup(article_id, db):
    """Supprime un article ET son PDF du disque dur"""
    article = db.query(Article).filter(Article.id == article_id).first()
    
    if not article:
        return False
    
    # Supprimer le PDF si il existe
    if article.pdf_path and os.path.exists(article.pdf_path):
        try:
            os.remove(article.pdf_path)
            print(f"Deleted PDF: {article.pdf_path}")
        except Exception as e:
            print(f"Warning: Could not delete PDF {article.pdf_path}: {e}")
    
    db.delete(article)
    db.commit()
    return True


def delete_session_with_cleanup(session_id, db):
    """Supprime une session ET tous ses articles ET PDFs associés"""
    session = db.query(SearchSession).filter(SearchSession.id == session_id).first()
    
    if not session:
        return False
    
    # Supprimer tous les articles de la session
    articles = db.query(Article).filter(Article.search_session_id == session_id).all()
    
    for article in articles:
        if article.pdf_path and os.path.exists(article.pdf_path):
            try:
                os.remove(article.pdf_path)
                print(f"Deleted PDF: {article.pdf_path}")
            except Exception as e:
                print(f"Warning: Could not delete PDF: {e}")
    
    # Supprimer la session et tous ses articles (cascade)
    db.query(Article).filter(Article.search_session_id == session_id).delete()
    db.delete(session)
    db.commit()
    
    return True
