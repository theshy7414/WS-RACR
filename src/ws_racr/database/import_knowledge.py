#!/usr/bin/env python3
"""
知識庫匯入工具
Import Knowledge Base to Database

功能:
1. 匯入 assembly_sop.yaml 到 sops 表
2. 匯入 system_overview.md 到 knowledge_fragments 表
3. 自動生成向量 embeddings (pgvector)
4. 建立 RAG 檢索索引

使用:
    python3 import_knowledge.py --all
    python3 import_knowledge.py --sop
    python3 import_knowledge.py --docs
"""
import sys
import yaml
import json
import argparse
from pathlib import Path
from datetime import datetime
from sqlalchemy import create_engine, text
import numpy as np

# 導入共享 embedding 模型
from ws_racr.database.embedding_singleton import get_embedding_model

# 資料庫連線
DB_URL = "postgresql+psycopg2://your_user:your_password@localhost/ragdb"

# Embedding 模型
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class KnowledgeImporter:
    """知識庫匯入器"""
    
    def __init__(self, db_url=DB_URL):
        """初始化"""
        self.engine = create_engine(db_url)
        self.embedding_model = get_embedding_model(EMBEDDING_MODEL)  # 使用共享實例
        print(f"✅ 已連接資料庫: {db_url}")
        print(f"✅ 已載入 Embedding 模型: {EMBEDDING_MODEL}")
    
    def generate_embedding(self, text):
        """生成文字向量"""
        embedding = self.embedding_model.encode(text)
        return embedding.tolist()
    
    def clear_table(self, table_name):
        """清空表格資料"""
        with self.engine.connect() as conn:
            result = conn.execute(text(f"SELECT COUNT(*) FROM {table_name}"))
            count = result.fetchone()[0]
            
            if count > 0:
                confirm = input(f"⚠️  {table_name} 表有 {count} 筆資料，確定要清空嗎? (y/N): ")
                if confirm.lower() == 'y':
                    conn.execute(text(f"DELETE FROM {table_name}"))
                    conn.commit()
                    print(f"✅ 已清空 {table_name} 表")
                else:
                    print(f"❌ 取消清空 {table_name} 表")
                    return False
        return True
    
    def import_sop(self, sop_file="assembly_sop.yaml", clear=False):
        """
        匯入 SOP 到資料庫
        
        策略:
        - 將整個 YAML 轉為結構化的 SOP fragments
        - 每個 phase 一筆記錄
        - 每個 step 一筆記錄
        - 錯誤處理、安全要求等也各一筆
        """
        print("\n" + "="*70)
        print("📥 匯入 SOP 知識庫")
        print("="*70)
        
        if clear and not self.clear_table('sops'):
            return
        
        # 讀取 YAML
        sop_path = Path(__file__).parent / sop_file
        if not sop_path.exists():
            print(f"❌ 找不到檔案: {sop_path}")
            return
        
        with open(sop_path, 'r', encoding='utf-8') as f:
            sop_data = yaml.safe_load(f)
        
        print(f"✅ 已讀取 SOP 檔案: {sop_path}")
        
        # 匯入主任務資訊
        self._import_main_task_info(sop_data)
        
        # 匯入各階段
        for phase_key in ['phase_1', 'phase_2', 'phase_3']:
            if phase_key in sop_data:
                self._import_phase(phase_key, sop_data[phase_key])
        
        print("\n✅ SOP 匯入完成！")
    
    def _import_main_task_info(self, sop_data):
        """匯入主任務資訊"""
        task_name = sop_data.get('task', 'unknown')
        description = sop_data.get('description', '')
        difficulty = sop_data.get('difficulty', '')
        estimated_time = sop_data.get('estimated_time_minutes', 0)
        safety_reqs = sop_data.get('safety_requirements', [])
        
        content = f"""
任務: {task_name}
描述: {description}
難度: {difficulty}
預估時間: {estimated_time} 分鐘

安全要求:
{chr(10).join(['- ' + req for req in safety_reqs])}
        """.strip()
        
        sop_id = f"sop_main_task_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        embedding = self.generate_embedding(content)
        
        with self.engine.connect() as conn:
            conn.execute(text("""
                INSERT INTO sops (sop_id, title, content, content_embedding, category, tags, metadata)
                VALUES (:sop_id, :title, :content, :embedding, :category, ARRAY['assembly', 'task_overview'], CAST(:metadata AS jsonb))
            """), {
                'sop_id': sop_id,
                'title': f"主任務: {task_name}",
                'content': content,
                'embedding': str(embedding),
                'category': 'main_task',
                'metadata': json.dumps({
                    'difficulty': difficulty,
                    'estimated_time_minutes': estimated_time
                })
            })
            conn.commit()
        
        print(f"  ✅ 已匯入主任務資訊")
    
    def _import_phase(self, phase_key, phase_data):
        """匯入階段資訊"""
        phase_name = phase_data.get('name', phase_key)
        phase_desc = phase_data.get('description', '')
        required_parts = phase_data.get('required_parts', [])
        
        # 階段概述
        parts_text = "\n".join([
            f"- {part['type']} × {part['quantity']}"
            for part in required_parts
        ])
        
        content = f"""
階段: {phase_name}
描述: {phase_desc}

所需零件:
{parts_text}
        """.strip()
        
        sop_id = f"sop_{phase_key}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        embedding = self.generate_embedding(content)
        
        with self.engine.connect() as conn:
            conn.execute(text("""
                INSERT INTO sops (sop_id, title, content, content_embedding, category, tags, metadata)
                VALUES (:sop_id, :title, :content, :embedding, :category, ARRAY['phase', 'assembly'], CAST(:metadata AS jsonb))
            """), {
                'sop_id': sop_id,
                'title': f"{phase_key.upper()}: {phase_name}",
                'content': content,
                'embedding': str(embedding),
                'category': phase_key,
                'metadata': json.dumps({'phase': phase_key})
            })
            conn.commit()
        
        print(f"  ✅ 已匯入階段: {phase_name}")
        
        # 匯入步驟
        steps = phase_data.get('steps', [])
        for step in steps:
            self._import_step(phase_key, step)
    
    def _import_step(self, phase_key, step_data):
        """匯入單一步驟"""
        step_id = step_data.get('id', 'unknown')
        step_name = step_data.get('name', '')
        step_action = step_data.get('action', '')
        step_desc = step_data.get('description', '')
        obj = step_data.get('object', '')
        tools = step_data.get('tools', [])
        robot_help = step_data.get('robot_help', [])
        preconditions = step_data.get('preconditions', [])
        
        # 組合步驟內容
        robot_help_text = ""
        if robot_help:
            robot_help_text = "\n機器人協助:\n" + "\n".join([
                f"- {help.get('type', 'unknown')}: {help.get('purpose', '')}" if isinstance(help, dict) else f"- {help}"
                for help in robot_help
            ])
        
        preconditions_text = ""
        if preconditions:
            preconditions_text = "\n前置條件:\n" + "\n".join([
                f"- {cond}" for cond in preconditions
            ])
        
        content = f"""
步驟 {step_id}: {step_name}
動作類型: {step_action}
描述: {step_desc}
物件: {obj}
工具: {', '.join(tools) if tools else '無'}
{robot_help_text}
{preconditions_text}
        """.strip()
        
        sop_id = f"sop_step_{phase_key}_{step_id}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        embedding = self.generate_embedding(content)
        
        with self.engine.connect() as conn:
            conn.execute(text("""
                INSERT INTO sops (sop_id, title, content, content_embedding, category, tags, metadata)
                VALUES (:sop_id, :title, :content, :embedding, :category, ARRAY[:tag1, :tag2, :tag3], CAST(:metadata AS jsonb))
            """), {
                'sop_id': sop_id,
                'title': f"步驟 {step_id}: {step_name}",
                'content': content,
                'embedding': str(embedding),
                'category': f'{phase_key}_step',
                'tag1': 'step',
                'tag2': step_action,
                'tag3': phase_key,
                'metadata': json.dumps({
                    'phase': phase_key,
                    'step_id': step_id,
                    'action': step_action
                })
            })
            conn.commit()
        
        print(f"    ✅ 已匯入步驟 {step_id}: {step_name}")
    
    def import_key_knowledge(self, json_file="key_knowledge.json", clear=False):
        """
        匯入關鍵知識 JSON 到 knowledge_fragments 表
        
        策略:
        - 從精簡的 JSON 提取關鍵實用知識
        - 每個主要類別一筆或多筆記錄
        - 產生向量便於 RAG 檢索
        """
        print("\n" + "="*70)
        print("📥 匯入關鍵知識庫 (JSON)")
        print("="*70)
        
        if clear and not self.clear_table('knowledge_fragments'):
            return
        
        # 讀取 JSON
        json_path = Path(__file__).parent / json_file
        if not json_path.exists():
            print(f"❌ 找不到檔案: {json_path}")
            return
        
        with open(json_path, 'r', encoding='utf-8') as f:
            knowledge = json.load(f)
        
        print(f"✅ 已讀取知識 JSON: {json_path}")
        
        # 匯入各類知識
        self._import_robot_assistance(knowledge.get('robot_assistance_types', []))
        self._import_human_gestures(knowledge.get('human_gestures', []))
        self._import_assembly_steps(knowledge.get('assembly_steps', []))
        self._import_robot_skills(knowledge.get('robot_skills', []))
        self._import_safety_rules(knowledge.get('safety_rules', []))
        self._import_common_scenarios(knowledge.get('common_scenarios', []))
        self._import_error_handling(knowledge.get('error_handling', []))
        
        print("\n✅ 關鍵知識匯入完成！")
    
    def _import_robot_assistance(self, assistance_types):
        """匯入機器人協助類型"""
        for assist in assistance_types:
            content = f"""
機器人協助類型: {assist['type']}
描述: {assist['description']}
範例: {assist.get('example', 'N/A')}
適用階段: {assist.get('phase', 'N/A')}
觸發條件: {assist.get('trigger', 'N/A')}
優先級: {assist.get('priority', '一般')}
            """.strip()
            
            fragment_id = f"knowledge_robot_assist_{assist['type']}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
            embedding = self.generate_embedding(content)
            
            tags_array = ['robot_assistance', assist['type'].lower()]
            tags_sql = "ARRAY[" + ",".join([f"'{tag}'" for tag in tags_array]) + "]"
            
            with self.engine.connect() as conn:
                conn.execute(text(f"""
                    INSERT INTO knowledge_fragments 
                    (fragment_id, fragment_type, title, content, content_embedding, 
                     source_type, confidence, tags, added_to_rag)
                    VALUES 
                    (:fragment_id, :fragment_type, :title, :content, :embedding,
                     :source_type, :confidence, {tags_sql}, :added_to_rag)
                """), {
                    'fragment_id': fragment_id,
                    'fragment_type': 'robot_assistance',
                    'title': f"機器人協助: {assist['type']}",
                    'content': content,
                    'embedding': str(embedding),
                    'source_type': 'key_knowledge_json',
                    'confidence': 1.0,
                    'added_to_rag': True
                })
                conn.commit()
            
            print(f"  ✅ 已匯入機器人協助: {assist['type']}")
    
    def _import_human_gestures(self, gestures):
        """匯入人類手勢知識"""
        for gesture in gestures:
            content = f"""
手勢: {gesture['gesture']}
意義: {gesture['meaning']}
機器人動作: {gesture['robot_action']}
觸發條件: {gesture.get('threshold', 'N/A')}
            """.strip()
            
            fragment_id = f"knowledge_gesture_{gesture['gesture']}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
            embedding = self.generate_embedding(content)
            
            tags_array = ['human_gesture', gesture['gesture']]
            tags_sql = "ARRAY[" + ",".join([f"'{tag}'" for tag in tags_array]) + "]"
            
            with self.engine.connect() as conn:
                conn.execute(text(f"""
                    INSERT INTO knowledge_fragments 
                    (fragment_id, fragment_type, title, content, content_embedding, 
                     source_type, confidence, tags, added_to_rag)
                    VALUES 
                    (:fragment_id, :fragment_type, :title, :content, :embedding,
                     :source_type, :confidence, {tags_sql}, :added_to_rag)
                """), {
                    'fragment_id': fragment_id,
                    'fragment_type': 'human_gesture',
                    'title': f"手勢識別: {gesture['gesture']}",
                    'content': content,
                    'embedding': str(embedding),
                    'source_type': 'key_knowledge_json',
                    'confidence': 1.0,
                    'added_to_rag': True
                })
                conn.commit()
            
            print(f"  ✅ 已匯入手勢: {gesture['gesture']}")
    
    def _import_assembly_steps(self, steps):
        """匯入組裝步驟知識"""
        for step in steps:
            content = f"""
步驟 {step['step_id']}: {step['name']}
描述: {step['description']}
階段: {step['phase']}
所需物件: {', '.join(step['required_objects'])}
機器人協助: {step['robot_help']}
            """.strip()
            
            fragment_id = f"knowledge_step_{step['name']}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
            embedding = self.generate_embedding(content)
            
            tags_array = ['assembly_step', step['name'], f"phase_{step['phase']}"]
            tags_sql = "ARRAY[" + ",".join([f"'{tag}'" for tag in tags_array]) + "]"
            
            with self.engine.connect() as conn:
                conn.execute(text(f"""
                    INSERT INTO knowledge_fragments 
                    (fragment_id, fragment_type, title, content, content_embedding, 
                     source_type, confidence, tags, added_to_rag)
                    VALUES 
                    (:fragment_id, :fragment_type, :title, :content, :embedding,
                     :source_type, :confidence, {tags_sql}, :added_to_rag)
                """), {
                    'fragment_id': fragment_id,
                    'fragment_type': 'assembly_step_info',
                    'title': f"步驟: {step['name']}",
                    'content': content,
                    'embedding': str(embedding),
                    'source_type': 'key_knowledge_json',
                    'confidence': 1.0,
                    'added_to_rag': True
                })
                conn.commit()
            
            print(f"  ✅ 已匯入步驟: {step['name']}")
    
    def _import_robot_skills(self, skills):
        """匯入機器人技能知識"""
        for skill in skills:
            content = f"""
技能: {skill['skill']}
描述: {skill['description']}
參數: {', '.join(skill['parameters'])}
使用場景: {skill.get('usage', 'N/A')}
安全注意: {skill.get('safety', 'N/A')}
優先級: {skill.get('priority', '一般')}
            """.strip()
            
            fragment_id = f"knowledge_skill_{skill['skill']}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
            embedding = self.generate_embedding(content)
            
            tags_array = ['robot_skill', skill['skill'].lower()]
            tags_sql = "ARRAY[" + ",".join([f"'{tag}'" for tag in tags_array]) + "]"
            
            with self.engine.connect() as conn:
                conn.execute(text(f"""
                    INSERT INTO knowledge_fragments 
                    (fragment_id, fragment_type, title, content, content_embedding, 
                     source_type, confidence, tags, added_to_rag)
                    VALUES 
                    (:fragment_id, :fragment_type, :title, :content, :embedding,
                     :source_type, :confidence, {tags_sql}, :added_to_rag)
                """), {
                    'fragment_id': fragment_id,
                    'fragment_type': 'robot_skill',
                    'title': f"機器人技能: {skill['skill']}",
                    'content': content,
                    'embedding': str(embedding),
                    'source_type': 'key_knowledge_json',
                    'confidence': 1.0,
                    'added_to_rag': True
                })
                conn.commit()
            
            print(f"  ✅ 已匯入技能: {skill['skill']}")
    
    def _import_safety_rules(self, rules):
        """匯入安全規則知識"""
        for rule in rules:
            content = f"""
安全規則: {rule['rule']}
優先級: {rule['priority']}
動作: {rule.get('action', 'N/A')}
參數: {rule.get('parameters', rule.get('method', 'N/A'))}
            """.strip()
            
            fragment_id = f"knowledge_safety_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
            embedding = self.generate_embedding(content)
            
            tags_array = ['safety', rule['priority']]
            tags_sql = "ARRAY[" + ",".join([f"'{tag}'" for tag in tags_array]) + "]"
            
            with self.engine.connect() as conn:
                conn.execute(text(f"""
                    INSERT INTO knowledge_fragments 
                    (fragment_id, fragment_type, title, content, content_embedding, 
                     source_type, confidence, tags, added_to_rag)
                    VALUES 
                    (:fragment_id, :fragment_type, :title, :content, :embedding,
                     :source_type, :confidence, {tags_sql}, :added_to_rag)
                """), {
                    'fragment_id': fragment_id,
                    'fragment_type': 'safety_rule',
                    'title': f"安全規則: {rule['rule'][:50]}",
                    'content': content,
                    'embedding': str(embedding),
                    'source_type': 'key_knowledge_json',
                    'confidence': 1.0,
                    'added_to_rag': True
                })
                conn.commit()
            
            print(f"  ✅ 已匯入安全規則")
    
    def _import_common_scenarios(self, scenarios):
        """匯入常見情境知識"""
        for scenario in scenarios:
            content = f"""
情境: {scenario['scenario']}
檢測方式: {scenario['detection']}
機器人動作: {scenario['robot_action']}
原因: {scenario.get('reason', 'N/A')}
時機: {scenario.get('timing', 'N/A')}
優先級: {scenario.get('priority', '一般')}
            """.strip()
            
            fragment_id = f"knowledge_scenario_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
            embedding = self.generate_embedding(content)
            
            tags_array = ['common_scenario', 'operational']
            tags_sql = "ARRAY[" + ",".join([f"'{tag}'" for tag in tags_array]) + "]"
            
            with self.engine.connect() as conn:
                conn.execute(text(f"""
                    INSERT INTO knowledge_fragments 
                    (fragment_id, fragment_type, title, content, content_embedding, 
                     source_type, confidence, tags, added_to_rag)
                    VALUES 
                    (:fragment_id, :fragment_type, :title, :content, :embedding,
                     :source_type, :confidence, {tags_sql}, :added_to_rag)
                """), {
                    'fragment_id': fragment_id,
                    'fragment_type': 'common_scenario',
                    'title': f"情境: {scenario['scenario'][:50]}",
                    'content': content,
                    'embedding': str(embedding),
                    'source_type': 'key_knowledge_json',
                    'confidence': 1.0,
                    'added_to_rag': True
                })
                conn.commit()
            
            print(f"  ✅ 已匯入情境: {scenario['scenario'][:40]}...")
    
    def _import_error_handling(self, errors):
        """匯入錯誤處理知識"""
        for error in errors:
            content = f"""
錯誤: {error['error']}
條件: {error.get('condition', error.get('threshold', 'N/A'))}
處理動作: {error['action']}
後備方案: {error.get('fallback', error.get('retry', 'N/A'))}
機器人動作: {error.get('robot_action', 'N/A')}
安全考量: {error.get('safety', 'N/A')}
範例: {', '.join(error.get('examples', []))}
            """.strip()
            
            fragment_id = f"knowledge_error_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
            embedding = self.generate_embedding(content)
            
            tags_array = ['error_handling', 'troubleshooting']
            tags_sql = "ARRAY[" + ",".join([f"'{tag}'" for tag in tags_array]) + "]"
            
            with self.engine.connect() as conn:
                conn.execute(text(f"""
                    INSERT INTO knowledge_fragments 
                    (fragment_id, fragment_type, title, content, content_embedding, 
                     source_type, confidence, tags, added_to_rag)
                    VALUES 
                    (:fragment_id, :fragment_type, :title, :content, :embedding,
                     :source_type, :confidence, {tags_sql}, :added_to_rag)
                """), {
                    'fragment_id': fragment_id,
                    'fragment_type': 'error_handling',
                    'title': f"錯誤處理: {error['error'][:50]}",
                    'content': content,
                    'embedding': str(embedding),
                    'source_type': 'key_knowledge_json',
                    'confidence': 1.0,
                    'added_to_rag': True
                })
                conn.commit()
            
            print(f"  ✅ 已匯入錯誤處理: {error['error'][:40]}...")
    
    def import_system_docs(self, doc_file="../../docs/system_overview.md", clear=False):
        """
        匯入系統文件到 knowledge_fragments 表 (只匯入關鍵實用部分)
        
        策略:
        - 按章節切分 markdown
        - 只匯入關鍵實用章節 (過濾掉技術細節、TODO、未實現功能)
        - 產生向量便於檢索
        """
        print("\n" + "="*70)
        print("📥 匯入系統文件知識庫 (關鍵部分)")
        print("="*70)
        
        if clear and not self.clear_table('knowledge_fragments'):
            return
        
        # 讀取 Markdown
        doc_path = Path(__file__).parent / doc_file
        if not doc_path.exists():
            print(f"❌ 找不到檔案: {doc_path}")
            return
        
        with open(doc_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        print(f"✅ 已讀取系統文件: {doc_path}")
        
        # 按 # 切分章節
        sections = self._split_markdown_by_sections(content)
        print(f"  共切分 {len(sections)} 個章節")
        
        # 過濾出關鍵章節
        key_sections = self._filter_key_sections(sections)
        print(f"  過濾後保留 {len(key_sections)} 個關鍵章節")
        
        # 匯入每個章節
        for section_title, section_content in key_sections:
            self._import_knowledge_fragment(section_title, section_content, 'system_documentation')
        
        print("\n✅ 系統文件匯入完成！")
    
    def _split_markdown_by_sections(self, content):
        """按主要章節切分 Markdown"""
        sections = []
        current_title = ""
        current_content = []
        
        for line in content.split('\n'):
            if line.startswith('# ') and not line.startswith('##'):
                # 新章節開始
                if current_title and current_content:
                    sections.append((current_title, '\n'.join(current_content)))
                current_title = line.replace('# ', '').strip()
                current_content = [line]
            else:
                current_content.append(line)
        
        # 最後一個章節
        if current_title and current_content:
            sections.append((current_title, '\n'.join(current_content)))
        
        return sections
    
    def _filter_key_sections(self, sections):
        """過濾出關鍵實用章節"""
        key_sections = []
        
        # 定義要保留的關鍵章節關鍵字
        keep_keywords = [
            '功能',      # 系統功能說明
            '使用',      # 使用方法
            '操作',      # 操作流程
            '手勢',      # 手勢識別
            '語音',      # 語音控制
            '協作',      # 人機協作
            '錯誤',      # 錯誤處理
            '安全',      # 安全須知
            'Overview',  # 概述
            'Usage',     # 使用
            'Feature',   # 特性
        ]
        
        # 定義要排除的關鍵字 (技術細節、未實現功能)
        skip_keywords = [
            'TODO',
            '⚠️',
            'Architecture',  # 技術架構
            'Implementation', # 實現細節
            'Developer',     # 開發者文檔
            'API',           # API 文檔
            '資料庫設計',     # 資料庫細節
            '模型訓練',       # 訓練細節
        ]
        
        for title, content in sections:
            # 檢查是否應該保留
            should_keep = any(keyword in title for keyword in keep_keywords)
            should_skip = any(keyword in title or keyword in content[:500] for keyword in skip_keywords)
            
            # 保留已實現的功能 (有 標記)
            if '✅' in content[:500] and 'IMPLEMENTED' in content[:500]:
                should_keep = True
            
            if should_keep and not should_skip:
                key_sections.append((title, content))
                print(f"    ✅ 保留: {title}")
            else:
                print(f"    ⏭️  跳過: {title}")
        
        return key_sections
    
    def _import_knowledge_fragment(self, title, content, fragment_type):
        """匯入知識片段"""
        fragment_id = f"knowledge_{fragment_type}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        embedding = self.generate_embedding(content)
        
        # 抽取標籤 (簡單版本)
        tags = []
        if '✅' in content or 'IMPLEMENTED' in content:
            tags.append('implemented')
        if '⚠️' in content or 'TODO' in content:
            tags.append('todo')
        if 'Camera' in content or 'camera' in content:
            tags.append('camera')
        if 'LLM' in content or 'llm' in content:
            tags.append('llm')
        if 'Robot' in content or 'robot' in content:
            tags.append('robot')
        
        # 準備 tags array
        tags_array = tags if tags else []
        tags_sql = "ARRAY[" + ",".join([f"'{tag}'" for tag in tags_array]) + "]" if tags_array else "ARRAY[]::text[]"
        
        with self.engine.connect() as conn:
            conn.execute(text(f"""
                INSERT INTO knowledge_fragments 
                (fragment_id, fragment_type, title, content, content_embedding, 
                 source_type, confidence, tags, added_to_rag)
                VALUES 
                (:fragment_id, :fragment_type, :title, :content, :embedding,
                 :source_type, :confidence, {tags_sql}, :added_to_rag)
            """), {
                'fragment_id': fragment_id,
                'fragment_type': fragment_type,
                'title': title[:200],  # 限制長度
                'content': content[:5000],  # 限制長度
                'embedding': str(embedding),
                'source_type': 'markdown_documentation',
                'confidence': 1.0,
                'added_to_rag': True
            })
            conn.commit()
        
        print(f"  ✅ 已匯入知識片段: {title[:50]}...")
    
    def show_statistics(self):
        """顯示匯入統計"""
        print("\n" + "="*70)
        print("📊 知識庫統計")
        print("="*70)
        
        with self.engine.connect() as conn:
            # SOPs
            result = conn.execute(text("""
                SELECT category, COUNT(*) as count
                FROM sops
                GROUP BY category
                ORDER BY category
            """))
            print("\n【SOPs 表】")
            total_sops = 0
            for row in result:
                print(f"  {row.category}: {row.count} 筆")
                total_sops += row.count
            print(f"  總計: {total_sops} 筆")
            
            # Knowledge Fragments
            result = conn.execute(text("""
                SELECT fragment_type, COUNT(*) as count
                FROM knowledge_fragments
                GROUP BY fragment_type
                ORDER BY fragment_type
            """))
            print("\n【Knowledge Fragments 表】")
            total_fragments = 0
            for row in result:
                print(f"  {row.fragment_type}: {row.count} 筆")
                total_fragments += row.count
            print(f"  總計: {total_fragments} 筆")


def main():
    parser = argparse.ArgumentParser(description='匯入知識庫到資料庫')
    parser.add_argument('--all', action='store_true', help='匯入所有知識 (SOP + 關鍵知識)')
    parser.add_argument('--sop', action='store_true', help='匯入 SOP')
    parser.add_argument('--key', action='store_true', help='匯入關鍵知識 JSON')
    parser.add_argument('--docs', action='store_true', help='匯入系統文件 (完整版)')
    parser.add_argument('--clear', action='store_true', help='匯入前清空表格')
    parser.add_argument('--stats', action='store_true', help='只顯示統計資訊')
    
    args = parser.parse_args()
    
    # 初始化匯入器
    importer = KnowledgeImporter()
    
    if args.stats:
        importer.show_statistics()
        return
    
    if args.all:
        importer.import_sop(clear=args.clear)
        importer.import_key_knowledge(clear=False)  # 第二次不清空
    elif args.sop:
        importer.import_sop(clear=args.clear)
    elif args.key:
        importer.import_key_knowledge(clear=args.clear)
    elif args.docs:
        importer.import_system_docs(clear=args.clear)
    else:
        print("請指定匯入選項: --all (推薦), --sop, --key, --docs, 或 --stats")
        parser.print_help()
        return
    
    # 顯示統計
    importer.show_statistics()


if __name__ == '__main__':
    main()
