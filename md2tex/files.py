import os
import subprocess
import shutil

def read_md_files(directory):
    # List all files with .md extension that don't start with "z-"
    md_files = [f for f in os.listdir(directory) if f.endswith('.md') and not f.startswith('z-')]
    md_files.sort()

    files_content = {}
    for file in md_files:
        with open(os.path.join(directory, file), 'r') as md_file:
            files_content[file] = md_file.read()

    return files_content

def prepare_tex_files(files_content, output_folder):
    include_commands = ""
    for filename, content in files_content.items():
        new_filename = os.path.splitext(filename)[0] + '.tex'
        with open(os.path.join(output_folder, new_filename), 'w') as tex_file:
            tex_file.write(content)

        # Use the output_folder variable in the \include command
        include_path = os.path.join(output_folder, os.path.splitext(filename)[0]).replace("\\", "/")  # Replace to ensure forward slashes
        include_commands += "\\include{" + include_path + "}\n"

    return include_commands

def generate_tex_file(template_path, output_path, include_commands):
    with open(template_path, 'r') as template_file:
        template_content = template_file.read()

    # Replace the placeholder with \include commands
    output_content = template_content.replace('{{chapters}}', include_commands)

    # Write to the output file
    with open(output_path, 'w') as output_file:
        output_file.write(output_content)

def convert_tex_to_pdf(main_file, tex_dir, aux_output_dir=".", pdf_output_dir="."):
    tex_file = os.path.join(tex_dir, main_file + ".tex")

    # Ensure the output directory exists
    os.makedirs(aux_output_dir, exist_ok=True)

    # Run pdflatex command with specified output directory for auxiliary files
    subprocess.run(["pdflatex", "-output-directory=" + aux_output_dir, tex_file], check=True)
    print(f"Successfully compiled {tex_file}.")

    # Change to the aux_output_dir for makeindex
    original_dir = os.getcwd()
    os.chdir(os.path.join(original_dir, aux_output_dir))
    idx_file = main_file + ".idx"
    subprocess.run(["makeindex", idx_file], check=True)
    print(f"Index generated successfully for {idx_file}.")
    os.chdir(original_dir)

    # Run pdflatex command with specified output directory for auxiliary files
    subprocess.run(["pdflatex", "-output-directory=" + aux_output_dir, tex_file], check=True)
    print(f"Successfully compiled {tex_file}.")

    # Move the PDF to the desired directory
    pdf_name = os.path.splitext(os.path.basename(tex_file))[0] + ".pdf"
    source_pdf = os.path.join(aux_output_dir, pdf_name)
    target_pdf = os.path.join(pdf_output_dir, pdf_name)

    shutil.move(source_pdf, target_pdf)
    print(f"Moved {pdf_name} to {pdf_output_dir}.")