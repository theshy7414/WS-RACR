"""Initial schema with task_edges and feedback tables

Revision ID: 001_initial
Revises: 
Create Date: 2025-10-30

This is the unified schema migration that consolidates:
- tasks table (with all Phase 3 enhancements)
- sops table (knowledge base)
- task_edges table (NEW: task graph relationships)
- feedback table (NEW: scoring and learning)

All indexes are managed here in one place.
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = '001_initial'
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    """
    Create all tables and indexes in unified way
    """
    
    # ========== 1. Enable pgvector extension ==========
    op.execute('CREATE EXTENSION IF NOT EXISTS vector')
    
    # ========== 2. Create tasks table ==========
    op.create_table(
        'tasks',
        sa.Column('id', sa.Integer(), primary_key=True),
        sa.Column('task_id', sa.String(100), unique=True, nullable=False),
        sa.Column('user_input', sa.Text(), nullable=False),
        sa.Column('task_type', sa.String(50), nullable=False),
        sa.Column('target_object', sa.String(100)),
        sa.Column('target_location', sa.String(100)),
        sa.Column('task_json', postgresql.JSONB(), nullable=False),
        sa.Column('task_embedding', postgresql.VECTOR(384)),
        sa.Column('plan_json', postgresql.JSONB()),
        sa.Column('telemetry', postgresql.JSONB()),
        sa.Column('status', sa.String(20), server_default='pending'),
        sa.Column('session_id', sa.String(100)),  # Phase 3: conversation tracking
        sa.Column('revision_of', sa.String(100)),  # Phase 3: plan revision tracking
        sa.Column('error_message', sa.Text()),
        sa.Column('created_at', sa.TIMESTAMP(), server_default=sa.func.now()),
        sa.Column('executed_at', sa.TIMESTAMP()),
        sa.Column('completed_at', sa.TIMESTAMP())
    )
    
    # Tasks indexes
    op.create_index('idx_tasks_status', 'tasks', ['status'])
    op.create_index('idx_tasks_created_at', 'tasks', ['created_at'], postgresql_using='btree', postgresql_ops={'created_at': 'DESC'})
    op.create_index('idx_tasks_executed_at', 'tasks', ['executed_at'], postgresql_using='btree', postgresql_ops={'executed_at': 'DESC'})
    op.create_index('idx_tasks_session', 'tasks', ['session_id'])
    op.create_index('idx_tasks_revision', 'tasks', ['revision_of'])
    op.create_index('idx_tasks_task_json', 'tasks', ['task_json'], postgresql_using='gin')
    op.create_index('idx_tasks_plan_json', 'tasks', ['plan_json'], postgresql_using='gin')
    
    # Try HNSW for vector index (fallback to IVFFlat if not supported)
    try:
        op.execute("""
            CREATE INDEX idx_tasks_embedding_hnsw ON tasks 
            USING hnsw (task_embedding vector_cosine_ops)
            WITH (m = 16, ef_construction = 64)
        """)
        print("✅ Created HNSW index for tasks.task_embedding")
    except Exception as e:
        print(f"⚠️  HNSW not supported, falling back to IVFFlat: {e}")
        op.execute("""
            CREATE INDEX idx_tasks_embedding_ivfflat ON tasks 
            USING ivfflat (task_embedding vector_cosine_ops)
            WITH (lists = 100)
        """)
        print("✅ Created IVFFlat index for tasks.task_embedding")
    
    # ========== 3. Create sops table ==========
    op.create_table(
        'sops',
        sa.Column('id', sa.Integer(), primary_key=True),
        sa.Column('sop_id', sa.String(100), unique=True, nullable=False),
        sa.Column('title', sa.Text(), nullable=False),
        sa.Column('content', sa.Text(), nullable=False),
        sa.Column('content_embedding', postgresql.VECTOR(384)),
        sa.Column('category', sa.String(50)),
        sa.Column('tags', postgresql.ARRAY(sa.String())),
        sa.Column('metadata', postgresql.JSONB()),
        sa.Column('created_at', sa.TIMESTAMP(), server_default=sa.func.now()),
        sa.Column('updated_at', sa.TIMESTAMP(), server_default=sa.func.now())
    )
    
    # SOPs indexes
    op.create_index('idx_sops_category', 'sops', ['category'])
    op.create_index('idx_sops_metadata', 'sops', ['metadata'], postgresql_using='gin')
    
    # Try HNSW for SOPs vector index
    try:
        op.execute("""
            CREATE INDEX idx_sops_embedding_hnsw ON sops 
            USING hnsw (content_embedding vector_cosine_ops)
            WITH (m = 16, ef_construction = 64)
        """)
        print("✅ Created HNSW index for sops.content_embedding")
    except Exception as e:
        print(f"⚠️  HNSW not supported for SOPs, falling back to IVFFlat: {e}")
        op.execute("""
            CREATE INDEX idx_sops_embedding_ivfflat ON sops 
            USING ivfflat (content_embedding vector_cosine_ops)
            WITH (lists = 100)
        """)
        print("✅ Created IVFFlat index for sops.content_embedding")
    
    # ========== 4. Create task_edges table (NEW) ==========
    op.create_table(
        'task_edges',
        sa.Column('id', sa.Integer(), primary_key=True),
        sa.Column('task_id', sa.String(100), nullable=False),
        sa.Column('parent_task_id', sa.String(100), nullable=False),
        sa.Column('relation', sa.String(50), nullable=False),  # 'revision', 'dependency', 'anchor', 'return_from'
        sa.Column('metadata', postgresql.JSONB()),
        sa.Column('created_at', sa.TIMESTAMP(), server_default=sa.func.now()),
        sa.ForeignKeyConstraint(['task_id'], ['tasks.task_id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['parent_task_id'], ['tasks.task_id'], ondelete='CASCADE')
    )
    
    # Task edges indexes
    op.create_index('idx_edges_task', 'task_edges', ['task_id'])
    op.create_index('idx_edges_parent', 'task_edges', ['parent_task_id'])
    op.create_index('idx_edges_relation', 'task_edges', ['relation'])
    
    # ========== 5. Create feedback table (NEW) ==========
    op.create_table(
        'feedback',
        sa.Column('id', sa.Integer(), primary_key=True),
        sa.Column('task_id', sa.String(100), nullable=False),
        sa.Column('score', sa.Float(), nullable=False),  # -1.0 to 1.0
        sa.Column('reason', sa.Text()),
        sa.Column('source', sa.String(50)),  # 'human', 'system', 'validator'
        sa.Column('metrics', postgresql.JSONB()),  # execution_time, success_rate, safety_score, etc.
        sa.Column('created_at', sa.TIMESTAMP(), server_default=sa.func.now()),
        sa.ForeignKeyConstraint(['task_id'], ['tasks.task_id'], ondelete='CASCADE')
    )
    
    # Feedback indexes
    op.create_index('idx_feedback_task', 'feedback', ['task_id'])
    op.create_index('idx_feedback_score', 'feedback', ['score'])
    op.create_index('idx_feedback_source', 'feedback', ['source'])
    
    print("✅ All tables and indexes created successfully!")


def downgrade() -> None:
    """
    Drop all tables in reverse order
    """
    op.drop_table('feedback')
    op.drop_table('task_edges')
    op.drop_table('sops')
    op.drop_table('tasks')
    op.execute('DROP EXTENSION IF EXISTS vector')
    print("✅ All tables dropped")
