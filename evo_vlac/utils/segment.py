import spacy
import re
import time

nlp = spacy.load("en_core_web_sm")

def extract_action_objects(instruction):
    """
    提取指令中的源对象、目标对象和动作
    支持更复杂的句式，包括：
    1. 复合名词短语 (e.g., "the red block on the left")
    2. 代词指代 (e.g., "pick up the apple and put it...")
    3. 多种动词同义词
    4. 连词连接的多个对象 (e.g., "block and cup")
    
    返回字典包含: source_obj, dest_obj, actions
    """
    doc = nlp(instruction)
    
    source_obj = None
    dest_obj = None
    actions = []
    
    # 扩展动词库
    pick_verbs = {'pick', 'take', 'grab', 'lift', 'get', 'grasp', 'clutch', 'seize', 'collect', 'retrieve'}
    place_verbs = {'insert', 'put', 'place', 'drop', 'set', 'lay', 'position', 'deposit', 'leave', 'store'}
    move_verbs = {'move', 'push', 'pull', 'slide', 'shift', 'transfer', 'carry'}
    
    # 遍历句子中的动词
    for token in doc:
        if token.pos_ == "VERB":
            actions.append(token.lemma_)
            
            # 1. 寻找源对象 (Source Object)
            # 通常是 pick/move 类动词的直接宾语 (dobj)
            if token.lemma_ in pick_verbs or token.lemma_ in move_verbs:
                for child in token.children:
                    if child.dep_ == "dobj":
                        # 如果是代词 (it, them)，且我们已经有了源对象，则跳过
                        if child.pos_ != "PRON":
                            # 检查是否有误附着在名词上的目标介词短语 (e.g. "Move block to corner")
                            # 需要检查 dobj 及其并列名词 (conj)
                            potential_dest_preps = []
                            
                            def check_dest_prep(t):
                                for c in t.children:
                                    if c.dep_ == 'prep' and c.text in ['to', 'into', 'towards']:
                                        potential_dest_preps.append(c)
                                    if c.dep_ == 'conj':
                                        check_dest_prep(c)
                            
                            check_dest_prep(child)
                            
                            if potential_dest_preps:
                                # 从源对象描述中排除该介词短语
                                source_obj = get_full_phrase(child, exclude_children=potential_dest_preps)
                                # 如果还没有目标对象，尝试从中提取 (使用第一个找到的)
                                if not dest_obj:
                                    prep = potential_dest_preps[0]
                                    for ggc in prep.children:
                                        if ggc.dep_ == 'pobj':
                                            dest_obj = get_full_phrase(ggc)
                            else:
                                source_obj = get_full_phrase(child)
                                
                        elif source_obj is None:
                            source_obj = child.text

            # 2. 寻找目标对象 (Destination Object)
            # 通常是 place/move 类动词的介词宾语 (pobj)
            if token.lemma_ in place_verbs or token.lemma_ in move_verbs:
                # 检查介词短语
                for child in token.children:
                    if child.dep_ == "prep":
                        for grandchild in child.children:
                            if grandchild.dep_ == "pobj":
                                dest_obj = get_full_phrase(grandchild)
                
                # 特殊情况：如果动词是 place/put，且没有找到源对象，
                # 有时直接宾语就是源对象 (e.g. "Place the apple in the box")
                if source_obj is None:
                    for child in token.children:
                        if child.dep_ == "dobj":
                            if child.pos_ != "PRON":
                                source_obj = get_full_phrase(child)
                            elif source_obj is None:
                                source_obj = child.text

    # 3. 兜底策略 (Fallback)
    if not source_obj or not dest_obj:
        valid_chunks = []
        for chunk in doc.noun_chunks:
            if chunk.root.pos_ != "PRON":
                valid_chunks.append(chunk.text)
        
        if not source_obj and len(valid_chunks) >= 1:
            source_obj = valid_chunks[0]
        
        if not dest_obj:
            if len(valid_chunks) >= 2:
                if valid_chunks[1] != source_obj:
                    dest_obj = valid_chunks[1]
            elif len(valid_chunks) == 1 and source_obj != valid_chunks[0]:
                 dest_obj = valid_chunks[0]
    
    return {
        'source_object': source_obj,
        'destination_object': dest_obj, 
        'actions': actions,
        'raw_instruction': instruction
    }

def get_full_phrase(token, exclude_children=None):
    """
    递归获取包含修饰词的完整短语
    exclude_children: 需要排除的子节点列表 (Token objects)
    """
    if not token:
        return ""
    if exclude_children is None:
        exclude_children = []
        
    indices = set()
    
    def collect_indices(t):
        if t in exclude_children:
            return
        indices.add(t.i)
        for child in t.children:
            if child in exclude_children:
                continue
                
            # 包含常见的修饰成分
            if child.dep_ in [
                'det', 'amod', 'compound', 'nummod', 'poss', 'case',
                'prep', 'pobj', 'pcomp',
                'relcl', 'acl', 'nsubj', 'mark',
                'punct', 'advmod'
            ]:
                collect_indices(child)
            elif child.dep_ == 'cc':
                collect_indices(child)
            elif child.dep_ == 'conj':
                # 仅跟随名词性并列词，避免错误包含动词
                if child.pos_ in ['NOUN', 'PROPN', 'ADJ', 'DET', 'NUM', 'PRON']:
                    collect_indices(child)
    
    collect_indices(token)
    
    sorted_indices = sorted(list(indices))
    words = [token.doc[i].text for i in sorted_indices]
    
    return ' '.join(words)

def decompose_instruction(instruction):
    """
    将复杂的长指令拆解为多个阶段的子指令，并尝试替换代词为具体的对象描述。
    """
    doc = nlp(instruction)
    
    splits = []
    current_start = 0
    
    for token in doc:
        is_split = False
        split_offset = 0 # 0 means split before token, 1 means split after token
        skip_token = False # Whether to include the split token in the next segment
        
        # 1. Explicit sequence words
        if token.text.lower() in ['then', 'afterwards', 'finally']:
            is_split = True
            skip_token = True
            
        # 2. Sentence terminators
        elif token.text in ['.', ';', '!', '?']:
            is_split = True
            split_offset = 1 # Split after the punctuation
            skip_token = True 
        
        # 3. "and" / "but" connecting verbs
        elif token.text.lower() in ['and', 'but']:
            if token.head.pos_ == "VERB":
                # Check if it connects to another verb (conj)
                has_verb_conj = False
                for child in token.head.children:
                    if child.dep_ == 'conj' and child.pos_ == 'VERB' and child.i > token.i:
                        has_verb_conj = True
                        break
                
                if has_verb_conj:
                    is_split = True
                    skip_token = True
        
        # 4. Comma handling
        elif token.text == ',':
            # Look ahead
            next_i = token.i + 1
            while next_i < len(doc) and doc[next_i].is_space:
                next_i += 1
            
            if next_i < len(doc):
                next_token = doc[next_i]
                # If comma is followed by a verb, or a sequence word, or "and" followed by verb
                if next_token.pos_ == "VERB" or next_token.text.lower() in ['then']:
                     is_split = True
                     split_offset = 1 
                elif next_token.text.lower() in ['and', 'but']:
                    # Check if 'and' connects verbs
                    if next_token.head.pos_ == "VERB":
                        has_verb_conj = False
                        for child in next_token.head.children:
                            if child.dep_ == 'conj' and child.pos_ == 'VERB' and child.i > next_token.i:
                                has_verb_conj = True
                                break
                        if has_verb_conj:
                            is_split = True
                            split_offset = 1
        
        if is_split:
            # Define the end of the current segment
            end_index = token.i + split_offset
            
            # Avoid empty splits or splits that are just punctuation
            if end_index > current_start:
                sub_text = doc[current_start:end_index].text.strip()
                if sub_text:
                    splits.append(sub_text)
            
            # Define start of next segment
            if skip_token:
                current_start = token.i + 1
            else:
                current_start = end_index
                
    # Add the last segment
    if current_start < len(doc):
        splits.append(doc[current_start:].text.strip())
        
    # Post-processing to clean up and resolve pronouns
    final_segments = []
    last_source_object = None
    
    for s in splits:
        # Clean up text
        clean_s = s.strip()
        while clean_s and (clean_s.lower().startswith("and") or clean_s.lower().startswith("then") or clean_s.startswith(",")):
            if clean_s.lower().startswith("and"):
                clean_s = clean_s[3:].strip()
            elif clean_s.lower().startswith("then"):
                clean_s = clean_s[4:].strip()
            elif clean_s.startswith(","):
                clean_s = clean_s[1:].strip()
        
        while clean_s and clean_s[-1] in ['.', ',', ';', '!', '?']:
            clean_s = clean_s[:-1].strip()
            
        if not clean_s:
            continue
            
        # Analyze current segment
        info = extract_action_objects(clean_s)
        current_source = info['source_object']
        
        # Check for pronouns in source object
        # Common pronouns: it, them, this, that, these, those
        pronouns = ['it', 'them', 'this', 'that', 'these', 'those']
        
        resolved_text = clean_s
        
        if current_source and current_source.lower() in pronouns:
            if last_source_object:
                # Replace pronoun with last object
                # Use regex to replace whole word to avoid partial matches
                pattern = re.compile(r'\b' + re.escape(current_source) + r'\b', re.IGNORECASE)
                resolved_text = pattern.sub(last_source_object, clean_s)
                # Update current source for next iteration (though usually we keep the resolved one)
                current_source = last_source_object
        
        elif current_source:
            # Update context if we have a valid noun phrase
            last_source_object = current_source
            
        final_segments.append(resolved_text)
            
    return final_segments


if __name__ == '__main__':
    instruction = "Put up the bowl and place it in the white storage box."
    print(f"Instruction: {instruction}")
    print("="*50)
    start_time = time.time()
    result = extract_action_objects(instruction)
    end_time = time.time()
    print(f"源对象: {result['source_object']}")
    print(f"目标对象: {result['destination_object']}")
    print(f"动作: {result['actions']}")
    print(f"推理时间:{end_time-start_time}")

    print("="*50)
    start_time = time.time()
    print("任务分阶段测试:")
    segments = decompose_instruction(instruction)
    end_time = time.time()
    for i, seg in enumerate(segments):
        print(f"Stage {i+1}: {seg}")
    print(f"推理时间:{end_time-start_time}")
