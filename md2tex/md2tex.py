import json
from files import *
from preprocess import *
from convert import *
from convert_formulas import *

def save_tokens_to_file(tokens, filename="tokens.txt"):
    with open(filename, "w") as f:
        for token in tokens:
            f.write(json.dumps(token, indent=4))
            f.write("\n" + "-"*40 + "\n")  # Divider between tokens for better readability

def process_tokens(tokens):
    for token in tokens:
        if token['type'] == 'text':
            token['content'] = escape_latex_special_chars(token['content'])
            token['content'] = convert_md_text_formatting(token['content'])
        elif token['type'] == 'inline_math' or token['type'] == 'block_math':
            token['content'] = replace_special_notation(token['content'])
            token['content'] = convert_md_formulas(token['content'], token['type'])
            token['content'] = wrap_equations_with_tag(token['content'])
        elif token['type'] == 'image':
            src = token['src'].replace('./img', '../img')
            width = token.get('width', None)
            number = token.get('number', '')
            caption = token.get('caption', '')
            latex_img = convert_html_image_to_latex(src, width, number + caption)
            token['content'] = latex_img
        elif token['type'] == 'mleq':
            token['content'] = process_multi_line_equations(token['content'])
    return tokens

if __name__ == "__main__":
    output_directory = "./tex/"
    files_content = read_md_files(".")

    tex_content = {}
    for filename, tex_file in files_content.items():
        tex_file = convert_md_chapters(tex_file)
        content_blocks = find_content_blocks(tex_file.split('\n'))

        tokenize_content(content_blocks, lambda tokens: process_tokens(tokens))
        final_content = reconstruct_from_tokenized_content(content_blocks)

        tex_content[filename] = final_content

    include_commands = prepare_tex_files(tex_content, output_directory)

    generate_tex_file('./md2tex/template.tex', './tex/00-book.tex', include_commands)
    convert_tex_to_pdf("00-book", tex_dir = "tex", aux_output_dir="output", pdf_output_dir=".")
    print("Generated content and output PDF!")