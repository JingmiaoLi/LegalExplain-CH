from app.schemas.legal import LegalArticle


LEGAL_ARTICLES = [
    LegalArticle(
        id="or_337",
        article_number="Art. 337 OR",
        title="Immediate termination for valid reasons",
        text=(
            "Either party may terminate the employment relationship immediately "
            "for valid reasons if continuation cannot reasonably be expected."
        ),
        source_url="https://www.fedlex.admin.ch/eli/cc/27/317_321_377/en",
        topics=["immediate_dismissal"],
    ),
    LegalArticle(
        id="or_336",
        article_number="Art. 336 OR",
        title="Abusive termination",
        text=(
            "Termination may be abusive if it is given for legally protected "
            "or improper reasons."
        ),
        source_url="https://www.fedlex.admin.ch/eli/cc/27/317_321_377/en",
        topics=["abusive_termination"],
    ),
    LegalArticle(
        id="or_335c",
        article_number="Art. 335c OR",
        title="Notice periods after probation",
        text=(
            "After the probation period, statutory notice periods depend on "
            "the year of service unless modified by agreement within legal limits."
        ),
        source_url="https://www.fedlex.admin.ch/eli/cc/27/317_321_377/en",
        topics=["notice_period"],
    ),
    LegalArticle(
        id="or_324a",
        article_number="Art. 324a OR",
        title="Salary during prevention from working",
        text=(
            "The employer may be obliged to continue paying salary for a limited "
            "period if the employee is prevented from working through no fault of their own."
        ),
        source_url="https://www.fedlex.admin.ch/eli/cc/27/317_321_377/en",
        topics=["salary_during_illness"],
    ),
    LegalArticle(
        id="or_340",
        article_number="Art. 340 OR",
        title="Non-compete undertaking",
        text=(
            "A non-compete undertaking may be valid only under specific statutory "
            "conditions, including written form and access to sensitive business information."
        ),
        source_url="https://www.fedlex.admin.ch/eli/cc/27/317_321_377/en",
        topics=["non_compete"],
    ),
]