-- Only raw SQL script to isolate from python code for creating tables in database.
CREATE TABLE queries(
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    query_text TEXT NOT NULL,
    q_status VARCHAR(20) DEFAULT pending, --updated to 'completed' or 'failed'
    created_at TIMESTAMP DEFAULT NOW(),
);

CREATE TABLE documents(
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    query_id UUID REFERENCES queries,
    url TEXT NOT NULL,
    raw_content TEXT,
    clean_text TEXT, --cleaned by Pandas
    fetched_at TIMESTAMP DEFAULT NOW(),
);

CREATE TABLE sources(
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    query_id UUID REFERENCES queries,
    url TEXT NOT NULL,
    snippet TEXT,
);

CREATE TABLE agent_actions(
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    query_id UUID REFERENCES queries,
    step_number INTEGER,
    action_type VARCHAR(50),  --'search', 'retrive', 'generate'
    tool_input TEXT,
    tool_output TEXT,
    created_at TIMESTAMP DEFAULT NOW(),
);
