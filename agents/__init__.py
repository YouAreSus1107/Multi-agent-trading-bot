"""
War-Room Bot -- Agents Package (Day Trading Architecture)

Analyst Team:
  - TechnicalAnalyst   (fast loop, 2 min)
  - FundamentalsAnalyst (fast loop, 2 min)
  - NewsAnalyst        (slow loop, 5 min)
  - SentimentAnalyst   (slow loop, 5 min)

Research Team:
  - ResearchTeam       (Bull/Bear debate)

Risk:
  - RiskManagerAgent   (position sizing, kill switches)

Assessment:
  - AssessmentAgent    (end-of-day reports)
"""

from agents.technical_analyst import TechnicalAnalyst
from agents.fundamentals_analyst import FundamentalsAnalyst
from agents.news_analyst import NewsAnalyst
from agents.sentiment_analyst import SentimentAnalyst
from agents.research_team import ResearchTeam
from agents.risk_agent import RiskManagerAgent
from agents.assessment_agent import AssessmentAgent
