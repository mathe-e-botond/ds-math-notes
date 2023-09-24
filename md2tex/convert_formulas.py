import re

def convert_md_formulas(content, math_type):
    if math_type == 'inline_math':
        # remove the $ delimiters and wrap with \( and \)
        return r'\(' + content[1:-1] + r'\)'
    elif math_type == 'block_math':
        # remove the $$ delimiters and wrap with \[ and \]
        return r'\[' + content[2:-2] + r'\]'
    else:
        return content

def wrap_equations_with_tag(content):
    # Check if the content contains \tag
    if r'\tag{' in content:
        # For block math wrapped in \[ \]
        if content.startswith(r'\[') and content.endswith(r'\]'):
            content = content[2:-2]  # Strip off the \[ \] delimiters
            content = r'\begin{equation}' + content.strip() + r'\end{equation}'

    return content

def replace_special_notation(content):
    content = content.replace(r'\R', r'\mathbb{R}')
    content = content.replace(r'\degree', r'^{\circ}')
    return content

def process_multi_line_equations(content):
    lines = content.strip().split('<br>')
    formatted_lines = []

    for line in lines:
        # Remove the mleq tags if they exist in the line
        line = line.replace('<!--mleq-->', '').replace('<!--/mleq-->', '').strip()

        segments = re.split(r'(\$.+?\$)', line)  # Split the line at inline math segments
        formatted_segments = []

        for segment in segments:
            if segment.startswith('$') and segment.endswith('$'):
                # Convert the inline math into equation part
                formatted_segments.append(segment[1:-1])
            elif segment.strip():
                # Only handle non-empty text
                formatted_segments.append(r'\text{' + segment.strip() + r'}')

        # Add the alignment character at the start and join the formatted segments for this line
        formatted_lines.append('& ' + ' '.join(formatted_segments) + r' \\')

    # Wrap in flalign* environment
    return r'\begin{flalign*}' + '\n' + '\n'.join(formatted_lines) + r' && \end{flalign*}' 