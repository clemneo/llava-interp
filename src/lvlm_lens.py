import torch
import json
from pathlib import Path
import base64
from io import BytesIO
from PIL import Image

def create_interactive_logit_lens(hidden_states, norm, lm_head, tokenizer, image, model_name, image_filename, prompt, save_folder = ".", image_size=336, patch_size=14, misc_text=""):
    # Tokenize the prompt
    input_ids = tokenizer.encode(prompt)
    
    # Find the image token and replace it with image tokens
    img_token_id = 32000  # The token ID for <img>
    img_token_count = (image_size // patch_size) ** 2  # 576 for 336x336 image with 14x14 patches
    
    token_labels = []
    for token_id in input_ids:
        if token_id == img_token_id:
            # One indexed because the HTML logic wants it that way
            token_labels.extend([f"<IMG{(i+1):03d}>" for i in range(img_token_count)])
        else:
            token_labels.append(tokenizer.decode([token_id]))
    
    # Exclude the input embedding layer if it's included
    num_layers = len(hidden_states)
    sequence_length = hidden_states[0].size(1)
    
    all_top_tokens = []
    
    for layer in range(num_layers):
        layer_hidden_states = hidden_states[layer]
        
        # Apply norm and lm_head
        normalized = norm(layer_hidden_states)
        logits = lm_head(normalized)
        
        # Get probabilities
        probs = torch.softmax(logits, dim=-1)
        
        # Get top 5 tokens and their probabilities for each position
        top_5_values, top_5_indices = torch.topk(probs, k=5, dim=-1)
        
        layer_top_tokens = []
        for pos in range(sequence_length):
            top_5_tokens = [tokenizer.decode(idx.item()) for idx in top_5_indices[0, pos]]
            top_5_probs = [f"{prob.item():.4f}" for prob in top_5_values[0, pos]]
            layer_top_tokens.append(list(zip(top_5_tokens, top_5_probs)))
        
        all_top_tokens.append(layer_top_tokens)
    
    # Process the image: central crop and resize
    img_w, img_h = image.size
    min_dim = min(img_w, img_h)
    left = (img_w - min_dim) / 2
    top = (img_h - min_dim) / 2
    right = (img_w + min_dim) / 2
    bottom = (img_h + min_dim) / 2
    image_cropped = image.crop((left, top, right, bottom))
    image_resized = image_cropped.resize((image_size, image_size), Image.LANCZOS)
    
    # Convert image to base64
    buffered = BytesIO()
    image_resized.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    
    # Generate HTML
    html_content = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Interactive Logit Lens</title>
    <style>
        body { margin: 0; padding: 0; font-family: Arial, sans-serif; }
        .container { display: flex; }
        .image-container { 
            flex: 0 0 auto; 
            margin: 20px; 
            position: relative;
            width: 336px; /* Set to match image width */
        }
        .highlight-box {
            position: absolute;
            border: 2px solid red;
            pointer-events: none;
            display: none;
        }
        .table-container { 
            flex: 1 1 auto;
            position: relative;
            max-height: 90vh;
            overflow: auto;
            margin: 20px;
        }
        table { 
            border-collapse: separate;
            border-spacing: 0;
        }
        th, td { 
            border: 1px solid #ddd; 
            padding: 8px; 
            text-align: center;
            min-width: 80px;
        }
        th { 
            background-color: #f2f2f2; 
            font-weight: bold;
        }
        .corner-header {
            position: sticky;
            top: 0;
            left: 0;
            z-index: 3;
            background-color: #f2f2f2;
        }
        .row-header {
            position: sticky;
            left: 0;
            z-index: 2;
            background-color: #f2f2f2;
        }
        .col-header {
            position: sticky;
            top: 0;
            z-index: 1;
            background-color: #f2f2f2;
        }
        #tooltip {
            display: none;
            position: fixed;
            background: white;
            border: 1px solid black;
            padding: 5px;
            z-index: 1000;
            pointer-events: none;
            max-width: 300px;
            font-size: 14px;
        }
        .highlighted-row {
            background-color: #ffff99;
        }
        .image-info {
            margin-top: 10px;
            font-size: 14px;
            width: 100%;
            word-wrap: break-word;
        }
        .prompt {
            font-weight: bold;
            margin-bottom: 5px;
        }
        .instructions {
            font-style: italic;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="image-container">
            <img src="data:image/png;base64,IMAGEPLACEHOLDER" alt="Input Image" style="width: 336px; height: 336px;">
            <div class="highlight-box"></div>
            <div class="image-info">
                <p class="prompt">Prompt: "PROMPTPLACEHOLDER"</p>
                <p class="instructions">Instructions: Click on image to lock the patch, click on image/table to unlock</p>
                <p>Info: MISCPLACEHOLDER</p>
            </div>
        </div>
        <div class="table-container">
            <table id="logitLens"></table>
        </div>
    </div>
    <div id="tooltip"></div>
<script>
    const data = DATAPLACEMENT;
    const tokenLabels = TOKENLABELSPLACEMENT;
    const tooltip = document.getElementById('tooltip');
    const highlightBox = document.querySelector('.highlight-box');
    const image = document.querySelector('.image-container img');
    const table = document.getElementById('logitLens');
    
    const imageSize = IMAGESIZEPLACEHOLDER;
    const patchSize = PATCHSIZEPLACEHOLDER;
    const gridSize = imageSize / patchSize;
    
    let isLocked = false;
    let highlightedRow = null;
    let lockedPatchIndex = null;
    
    // Create corner header
    const cornerHeader = table.createTHead().insertRow();
    cornerHeader.insertCell().textContent = 'Token/Layer';
    cornerHeader.cells[0].classList.add('corner-header');
    
    // Create layer headers
    for (let i = 0; i < data.length; i++) {
        const th = document.createElement('th');
        th.textContent = `Layer ${i + 1}`;
        th.classList.add('col-header');
        cornerHeader.appendChild(th);
    }
    
    // Create rows with token labels
    for (let pos = 0; pos < tokenLabels.length; pos++) {
        const row = table.insertRow();
        const rowHeader = row.insertCell();
        rowHeader.textContent = tokenLabels[pos];
        rowHeader.classList.add('row-header');
        
        for (let layer = 0; layer < data.length; layer++) {
            const cell = row.insertCell();
            const topToken = data[layer][pos][0][0];
            cell.textContent = topToken;
            
            cell.addEventListener('mouseover', (e) => {
                if (!isLocked) {
                    showTooltip(e, layer, pos, false);
                }
            });
            cell.addEventListener('mousemove', updateTooltipPosition);
            cell.addEventListener('mouseout', () => {
                if (!isLocked) {
                    hideTooltip();
                }
            });
        }
    }

    function showTooltip(e, layer, pos, shouldScroll = false) {
        tooltip.innerHTML = data[layer][pos].map(([token, prob]) => `${token}: ${prob}`).join('<br>');
        tooltip.style.display = 'block';
        updateTooltipPosition(e);
        
        if (tokenLabels[pos].startsWith('<IMG')) {
            const patchIndex = parseInt(tokenLabels[pos].slice(4, 7));
            highlightImagePatch(patchIndex);
            highlightTableRow(pos, shouldScroll);
        } else {
            highlightBox.style.display = 'none';
            unhighlightTableRow();
        }
    }

    function hideTooltip() {
        tooltip.style.display = 'none';
        if (!isLocked) {
            highlightBox.style.display = 'none';
            unhighlightTableRow();
        }
    }

    function updateTooltipPosition(e) {
        const tooltipRect = tooltip.getBoundingClientRect();
        const viewportWidth = window.innerWidth;
        const viewportHeight = window.innerHeight;

        let x = e.clientX + 10;
        let y = e.clientY + 10;

        if (x + tooltipRect.width > viewportWidth) {
            x = e.clientX - tooltipRect.width - 10;
        }

        if (y + tooltipRect.height > viewportHeight) {
            y = e.clientY - tooltipRect.height - 10;
        }

        x = Math.max(0, x);
        y = Math.max(0, y);

        tooltip.style.left = x + 'px';
        tooltip.style.top = y + 'px';
    }
    
    function highlightImagePatch(patchIndex) {
        const scaleFactor = image.width / imageSize;
        const row = Math.floor((patchIndex - 1) / gridSize);
        const col = (patchIndex - 1) % gridSize;
        
        const left = col * patchSize * scaleFactor;
        const top = row * patchSize * scaleFactor;
        const size = patchSize * scaleFactor;
        
        highlightBox.style.left = `${left}px`;
        highlightBox.style.top = `${top}px`;
        highlightBox.style.width = `${size}px`;
        highlightBox.style.height = `${size}px`;
        highlightBox.style.display = 'block';
    }

    function highlightTableRow(rowIndex, shouldScroll = false) {
        if (highlightedRow) {
            highlightedRow.classList.remove('highlighted-row');
        }
        highlightedRow = table.rows[rowIndex + 1];  // +1 to account for header row
        highlightedRow.classList.add('highlighted-row');
        if (shouldScroll) {
            highlightedRow.scrollIntoView({ behavior: 'smooth', block: 'center' });
        }
    }

    function unhighlightTableRow() {
        if (highlightedRow) {
            highlightedRow.classList.remove('highlighted-row');
            highlightedRow = null;
        }
    }

    image.addEventListener('mousemove', (e) => {
        if (!isLocked) {
            const patchIndex = getPatchIndexFromMouseEvent(e);
            highlightImagePatch(patchIndex);
            const tokenIndex = getTokenIndexFromPatchIndex(patchIndex);
            if (tokenIndex !== -1) {
                showTooltip(e, 0, tokenIndex, true);
            }
        }
    });

    image.addEventListener('mouseout', () => {
        if (!isLocked) {
            hideTooltip();
        }
    });

    image.addEventListener('click', (e) => {
        isLocked = !isLocked;
        if (isLocked) {
            lockedPatchIndex = getPatchIndexFromMouseEvent(e);
            highlightImagePatch(lockedPatchIndex);
            const tokenIndex = getTokenIndexFromPatchIndex(lockedPatchIndex);
            if (tokenIndex !== -1) {
                highlightTableRow(tokenIndex, true);
            }
        } else {
            lockedPatchIndex = null;
            hideTooltip();
        }
    });

    table.addEventListener('click', () => {
        isLocked = false;
        lockedPatchIndex = null;
        hideTooltip();
    });

    function getPatchIndexFromMouseEvent(e) {
        const rect = image.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const y = e.clientY - rect.top;
        const patchX = Math.floor(x / (image.width / gridSize));
        const patchY = Math.floor(y / (image.height / gridSize));
        return patchY * gridSize + patchX + 1;
    }

    function getTokenIndexFromPatchIndex(patchIndex) {
        return tokenLabels.findIndex(label => label === `<IMG${patchIndex.toString().padStart(3, '0')}>`);
    }
</script>
</body>
</html>
    """
    
    # Replace placeholders
    html_content = html_content.replace('IMAGEPLACEHOLDER', img_str)
    html_content = html_content.replace('DATAPLACEMENT', json.dumps(all_top_tokens))
    html_content = html_content.replace('TOKENLABELSPLACEMENT', json.dumps(token_labels))
    html_content = html_content.replace('IMAGESIZEPLACEHOLDER', str(image_size))
    html_content = html_content.replace('PATCHSIZEPLACEHOLDER', str(patch_size))
    html_content = html_content.replace('PROMPTPLACEHOLDER', prompt)  # Add this line
    html_content = html_content.replace('MISCPLACEHOLDER', misc_text)  # Add this line

    
    # Create filename using model name and image filename
    output_filename = f"{model_name}_{Path(image_filename).stem}_logit_lens.html"
    
    # Join save folder and filename
    output_path = Path(save_folder) / output_filename

    # Write to file
    with open(output_path, 'w') as f:
        f.write(html_content)
    
    print(f"Interactive logit lens HTML has been saved to: {output_path}")

