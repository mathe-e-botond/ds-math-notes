import re

def find_content_blocks(lines):
    content_blocks = []

    # Tokenize patterns
    patterns = {
        'ordered_list': r'^\s*\d+\. (.*)$',
        'unordered_list': r'^\s*[\*-] (.*)$'
    }

    line_num = 0
    prev_block = None
    while line_num < len(lines):
        current_block = {}

        line = lines[line_num]
        line = line.rstrip()  # Remove trailing whitespace
        for block_type, pattern in patterns.items():
            match = re.match(pattern, line)
            if match:
                first_line = match.group(1)
                current_block["type"] = block_type
                break
        
        if not match:
            current_block = {"type": "text", "content": line}
        else:
            child_lines = [first_line.lstrip()]
            while line_num + 1 < len(lines):
                next_line = lines[line_num + 1]

                indentation = len(next_line) - len(next_line.lstrip())

                # Determine the indentation level (number of spaces)
                indentation = len(next_line) - len(next_line.lstrip())

                if indentation == 4:
                    line_num += 1
                    child_lines.append(next_line[4:])
                else:
                    break

            current_block['content'] = find_content_blocks(child_lines)
        
        if prev_block != None and current_block["type"] == prev_block["type"]:
            if current_block["type"] == 'text':
                prev_block["content"] += '\n' + current_block["content"]
            else:
                prev_block["content"].extend(current_block["content"])
        else:
            content_blocks.append(current_block)
            prev_block = current_block

        line_num += 1

    return content_blocks

def tokenize_content(content_blocks, callback):
    # Tokenize patterns
    patterns = {
        'mleq': r'<!--mleq-->(.*?)<!--/mleq-->',
        'image': r'<p align="(?P<align>\w+)">\s*<img src="(?P<src>.+?)"( width="(?P<width>\d+)")?>\s*(?:<br><b>(?P<number>Figure .+?: )</b>(?P<caption>.+?))?\s*</p>',  
        'block_math': r'(?<!\$)\$\$(.+?)\$\$(?!\$)', 
        'inline_math': r'(?<!\$)\$(.+?)\$(?!\$)', 
        'heading': r'\\(chapter|section|subsection)\{.+?\}',
        'text': r'.+?(?=[<\$\{]|\\|$)' # Explicitly avoiding '\'
    }

    for content_block in content_blocks:
        if content_block["type"]  == 'ordered_list' or content_block["type"]  == 'unordered_list':
            tokenize_content(content_block["content"], callback)
        elif content_block["type"]  == 'text':
            tokens = []
            content = content_block["content"]
            pos = 0
            while pos < len(content):
                match = None
                for token_type, pattern in patterns.items():
                    match = re.match(pattern, content[pos:], flags=re.DOTALL)
                    if match:
                        token_content = match.group(0)
                        
                        if token_type == 'heading':
                            level_match = re.match(r'\\(chapter|section|subsection)', token_content)
                            level = level_match.group(1) if level_match else None
                            tokens.append({'type': token_type, 'content': token_content, 'level': level})
                        elif token_type == 'image':
                            tokens.append({'type': token_type, 'content': token_content, **match.groupdict()})
                        else:
                            tokens.append({'type': token_type, 'content': token_content})
                            
                        pos += len(token_content)
                        break

                if not match:
                    pos += 1
            content_block["tokens"] = callback(tokens)

def reconstruct_from_tokenized_content(content_blocks, indent_level=0):
    latex_content = ""

    first_block = True
    for block in content_blocks:
        if block["type"] == "text":
            # Append the text content with appropriate indentation
            if not first_block:
                latex_content += " " * (indent_level * 4)
            latex_content += ''.join(token['content'] for token in block["tokens"]) + "\n"
        elif block["type"] == "ordered_list":
            # Add the ordered list with indentation
            latex_content += " " * (indent_level * 4) + "\\begin{enumerate}\n"
            for item in block["content"]:
                # Recursively convert each list item
                latex_content += " " * ((indent_level + 1) * 4)
                if item["type"] == "text":
                    latex_content += "\\item "
                latex_content += reconstruct_from_tokenized_content([item], indent_level + 2)
            latex_content += " " * (indent_level * 4) + "\\end{enumerate}\n"
        elif block["type"] == "unordered_list":
            # Add the unordered list with indentation
            latex_content += " " * (indent_level * 4) + "\\begin{itemize}\n"
            for item in block["content"]:
                # Recursively convert each list item
                latex_content += " " * ((indent_level + 1) * 4)
                if item["type"] == "text":
                    latex_content += "\\item "
                latex_content += reconstruct_from_tokenized_content([item], indent_level + 2)
            latex_content += " " * (indent_level * 4) + "\\end{itemize}\n"
        first_block = False

    return latex_content