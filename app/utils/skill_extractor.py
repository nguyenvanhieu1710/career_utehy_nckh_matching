# app/skill_extractor.py

SKILLS = [
    # Programming Languages
    "python", "java", "javascript", "typescript", "c++", "c#", "php", "ruby", "go", "rust", "swift",
    "html", "css", "sql", "r", "matlab",
    
    # Frameworks & Libraries
    "react", "vue", "angular", "nodejs", "express", "django", "flask", "fastapi", "spring", "laravel",
    "tensorflow", "pytorch", "keras", "scikit-learn", "pandas", "numpy", "matplotlib", "seaborn",
    
    # Databases
    "mysql", "postgresql", "mongodb", "redis", "sqlite", "oracle", "elasticsearch",
    
    # Cloud & DevOps
    "aws", "azure", "gcp", "docker", "kubernetes", "jenkins", "gitlab", "terraform", "ansible",
    "ci/cd", "microservices", "serverless",
    
    # Tools & Technologies
    "git", "github", "bitbucket", "jira", "confluence", "slack", "api", "rest", "graphql", "webhook",
    "linux", "ubuntu", "windows", "macos", "bash", "powershell", "shell",
    
    # AI/ML/Data Science
    "machine learning", "deep learning", "nlp", "computer vision", "data science", "analytics",
    "artificial intelligence", "neural network", "algorithm", "data mining", "statistics",
    
    # Frontend/UI
    "html5", "css3", "sass", "less", "tailwind", "bootstrap", "jquery", "webpack", "vite",
    "responsive design", "ui/ux", "figma", "sketch", "adobe xd",
    
    # Backend/Architecture
    "microservices", "rest api", "graphql", "websockets", "oauth", "jwt", "authentication",
    "authorization", "security", "encryption", "caching", "load balancing",
    
    # Testing
    "jest", "cypress", "selenium", "unit testing", "integration testing", "e2e testing", "tdd", "bdd",
    
    # Project Management
    "agile", "scrum", "kanban", "waterfall", "product management", "stakeholder management",
    "roadmap", "user research", "a/b testing", "analytics", "metrics", "kpi",
    
    # General Skills
    "leadership", "communication", "teamwork", "problem solving", "critical thinking", "creativity",
    "innovation", "strategy", "planning", "organization", "time management", "project management"
]

def extract_skills(text: str):
    text = text.lower()
    found_skills = []
    
    for skill in SKILLS:
        # Check for whole word matches to avoid partial matches
        import re
        pattern = r'\b' + re.escape(skill) + r'\b'
        if re.search(pattern, text):
            found_skills.append(skill)
    
    return found_skills
