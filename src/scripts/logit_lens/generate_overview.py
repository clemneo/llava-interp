import os
import argparse
import re
import json

def generate_cover_html(input_dir, output_dir):
    # Get all HTML files in the input directory
    html_files = [f for f in os.listdir(input_dir) if f.endswith('.html') and f != 'index.html']
    
    # Sort the files to ensure consistent ordering
    html_files.sort()

    # Extract val_XXXXXXXX part from filenames
    file_labels = [re.search(r'(val_\d+)', f).group(1) if re.search(r'(val_\d+)', f) else f for f in html_files]

    # Create a list of dictionaries containing file info
    file_info = [{"file": f, "label": l} for f, l in zip(html_files, file_labels)]

    # HTML template
    html_template = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>HTML File Viewer</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 20px;
        }
        #search {
            width: 100%;
            padding: 10px;
            margin-bottom: 20px;
        }
        #pagination {
            margin-bottom: 20px;
        }
        #file-list {
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
        }
        .file-button {
            padding: 10px;
            background-color: #f0f0f0;
            border: none;
            cursor: pointer;
        }
        .file-button.active {
            background-color: #ddd;
        }
        #file-viewer {
            width: 100%;
            height: 600px;
            border: 1px solid #ddd;
        }
    </style>
</head>
<body>
    <input type="text" id="search" placeholder="Search files...">
    <div id="pagination"></div>
    <div id="file-list"></div>
    <iframe id="file-viewer"></iframe>

    <script>
        const files = {file_info};
        const itemsPerPage = 20;
        let currentPage = 1;
        let filteredFiles = [...files];

        function displayFiles() {
            const fileList = document.getElementById('file-list');
            fileList.innerHTML = '';
            const start = (currentPage - 1) * itemsPerPage;
            const end = start + itemsPerPage;
            const pageFiles = filteredFiles.slice(start, end);

            pageFiles.forEach(file => {
                const button = document.createElement('button');
                button.textContent = file.label;
                button.className = 'file-button';
                button.onclick = () => loadFile(file.file);
                fileList.appendChild(button);
            });

            updatePagination();
        }

        function updatePagination() {
            const pageCount = Math.ceil(filteredFiles.length / itemsPerPage);
            const pagination = document.getElementById('pagination');
            pagination.innerHTML = '';

            for (let i = 1; i <= pageCount; i++) {
                const button = document.createElement('button');
                button.textContent = i;
                button.onclick = () => {
                    currentPage = i;
                    displayFiles();
                };
                pagination.appendChild(button);
            }
        }

        function loadFile(fileName) {
            const viewer = document.getElementById('file-viewer');
            viewer.src = fileName;
            
            // Update active button
            document.querySelectorAll('.file-button').forEach(btn => {
                btn.classList.remove('active');
                if (btn.textContent === fileName) {
                    btn.classList.add('active');
                }
            });
        }

        function searchFiles() {
            const searchTerm = document.getElementById('search').value.toLowerCase();
            filteredFiles = files.filter(file => 
                file.label.toLowerCase().includes(searchTerm) || 
                file.file.toLowerCase().includes(searchTerm)
            );
            currentPage = 1;
            displayFiles();
        }

        document.getElementById('search').addEventListener('input', searchFiles);

        // Initial display
        displayFiles();
    </script>
</body>
</html>
    '''

    # Replace the placeholder with the actual file info
    html_content = html_template.replace('{file_info}', json.dumps(file_info))

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Write the HTML content to index.html in the output directory
    output_file = os.path.join(output_dir, 'index.html')
    with open(output_file, 'w') as f:
        f.write(html_content)

    print(f"Generated {output_file} with links to {len(html_files)} HTML files.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a cover HTML file with links to HTML files in a directory.")
    parser.add_argument("input_dir", help="Directory containing the HTML files")
    parser.add_argument("output_dir", help="Directory where the index.html will be saved")
    args = parser.parse_args()

    generate_cover_html(args.input_dir, args.output_dir)