#!/usr/bin/env python3
"""
MLM-based caveman compression using RoBERTa.
Removes highly predictable tokens based on masked language model probabilities.
No LLM API required - uses local RoBERTa model for deterministic compression.

Probability thresholds:
  P >= 1e-3:  Conservative (16% reduction, 98% accuracy)
  P >= 1e-4:  Moderate (19% reduction, 97% accuracy)
  P >= 1e-5:  Balanced (32% reduction, 92% accuracy) [DEFAULT]
  P >= 1e-6:  Aggressive (54% reduction, 83% accuracy)
"""

import sys
import argparse
from pathlib import Path

try:
    import torch
    import torch.nn.functional as F
    from transformers import RobertaForMaskedLM, RobertaTokenizer
    import spacy
except ImportError:
    print("Error: Required packages not installed. Install with:", file=sys.stderr)
    print("  pip install torch transformers spacy", file=sys.stderr)
    print("  python -m spacy download en_core_web_sm", file=sys.stderr)
    sys.exit(1)

# Model cache
_roberta_model = None
_roberta_tokenizer = None
_device = None
_nlp = None

def get_models():
    """Load or retrieve cached models"""
    global _roberta_model, _roberta_tokenizer, _device, _nlp

    if _roberta_model is None:
        print("Loading models (this may take a moment)...", file=sys.stderr)
        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _roberta_tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        _roberta_model = RobertaForMaskedLM.from_pretrained('roberta-base').to(_device)
        _roberta_model.eval()
        _nlp = spacy.load('en_core_web_sm')
        print("Models loaded.", file=sys.stderr)

    return _roberta_model, _roberta_tokenizer, _device, _nlp

def count_tokens(text):
    """Estimate token count: characters / 4"""
    return len(text.strip()) // 4

def get_mlm_probability(model, tokenizer, device, sentence, word_idx):
    """Get MLM probability for a specific word in a sentence.

    Returns the probability P(word | context) that RoBERTa assigns to the
    original word appearing at the masked position.
    """
    words = sentence.split()
    if word_idx >= len(words):
        return 0.0

    target_word = words[word_idx]
    masked_words = words.copy()
    masked_words[word_idx] = tokenizer.mask_token
    masked_sentence = " ".join(masked_words)

    try:
        inputs = tokenizer(masked_sentence, return_tensors="pt", max_length=512, truncation=True).to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits

        mask_token_index = (inputs.input_ids == tokenizer.mask_token_id).nonzero(as_tuple=True)[1]
        if len(mask_token_index) == 0:
            return 0.0

        mask_token_logits = logits[0, mask_token_index[0], :]
        probs = F.softmax(mask_token_logits, dim=0)

        # Tokenize the target word to get its token ID(s)
        # Handle case where word might tokenize to multiple subwords
        target_tokens = tokenizer.encode(target_word, add_special_tokens=False)
        if len(target_tokens) == 0:
            return 0.0

        # Use the first subword token's probability as approximation
        target_token_id = target_tokens[0]
        prob = float(probs[target_token_id].item())

        return prob

    except Exception:
        return 0.0

def compress_text(text, prob_threshold=1e-5, no_adjacent_removal=False, protect_ner=True):
    """
    Apply MLM-based compression by removing words whose predictability exceeds threshold.

    Words with P(word | context) >= prob_threshold are considered highly predictable
    and are removed. Lower thresholds = more aggressive compression.

    Args:
        text: Input text to compress
        prob_threshold: Remove words with MLM probability >= this value (default: 1e-5)
            1e-3: Conservative (16% reduction)
            1e-4: Moderate (19% reduction)
            1e-5: Balanced (32% reduction) [DEFAULT]
            1e-6: Aggressive (54% reduction)
        no_adjacent_removal: If True, never remove two adjacent words. When both
            exceed threshold, only remove the one with higher probability.
        protect_ner: If True, never remove named entities (PERSON, ORG, GPE, DATE,
            MONEY, PERCENT, TIME, QUANTITY, CARDINAL, ORDINAL).

    Returns:
        Compressed text
    """
    roberta_model, roberta_tokenizer, device, nlp = get_models()

    # Protected NER labels - these carry important factual information
    PROTECTED_NER = {'PERSON', 'ORG', 'GPE', 'DATE', 'MONEY', 'PERCENT',
                     'TIME', 'QUANTITY', 'CARDINAL', 'ORDINAL', 'LOC', 'FAC'}

    # Split into sentences for better MLM context
    doc = nlp(text)
    sentences = [sent.text.strip() for sent in doc.sents]

    # Track which words to remove based on probability threshold
    words_to_remove = set()

    # Track protected word positions (from NER)
    protected_positions = set()

    for sent_idx, sentence in enumerate(sentences):
        # Parse sentence
        sent_doc = nlp(sentence)
        words = [token.text for token in sent_doc if not token.is_punct and not token.is_space]

        if len(words) == 0:
            continue

        # If NER protection enabled, find protected positions
        if protect_ner:
            # Map character positions to word indices
            word_tokens = [token for token in sent_doc if not token.is_punct and not token.is_space]
            for ent in sent_doc.ents:
                if ent.label_ in PROTECTED_NER:
                    # Find which word indices overlap with this entity
                    for word_idx, token in enumerate(word_tokens):
                        if token.idx >= ent.start_char and token.idx < ent.end_char:
                            protected_positions.add((sent_idx, word_idx))
                        elif token.idx + len(token.text) > ent.start_char and token.idx < ent.end_char:
                            protected_positions.add((sent_idx, word_idx))

        # Get MLM probabilities for each word in this sentence
        word_probs = []
        for word_idx, word in enumerate(words):
            prob = get_mlm_probability(roberta_model, roberta_tokenizer, device, sentence, word_idx)
            word_probs.append((sent_idx, word_idx, prob))

        if no_adjacent_removal:
            # Sort by probability (highest first) and greedily select non-adjacent removals
            candidates = [(s, w, p) for s, w, p in word_probs if p >= prob_threshold]
            candidates.sort(key=lambda x: x[2], reverse=True)

            removed_indices = set()
            for sent_i, word_i, prob in candidates:
                # Skip if this is a protected NER position
                if protect_ner and (sent_i, word_i) in protected_positions:
                    continue

                # Check if adjacent word was already removed
                prev_removed = (sent_i, word_i - 1) in removed_indices
                next_removed = (sent_i, word_i + 1) in removed_indices

                if not prev_removed and not next_removed:
                    removed_indices.add((sent_i, word_i))
                    words_to_remove.add((sent_i, word_i))
        else:
            # Original behavior: remove all words exceeding threshold
            for sent_i, word_i, prob in word_probs:
                if prob >= prob_threshold:
                    # Skip if this is a protected NER position
                    if protect_ner and (sent_i, word_i) in protected_positions:
                        continue
                    words_to_remove.add((sent_i, word_i))

    # Reconstruct sentences without removed words
    compressed_sentences = []
    for sent_idx, sentence in enumerate(sentences):
        sent_doc = nlp(sentence)
        words = [token.text for token in sent_doc if not token.is_punct and not token.is_space]

        if len(words) == 0:
            compressed_sentences.append(sentence)
            continue

        # Keep words that are not in the removal set
        remaining_words = [w for word_idx, w in enumerate(words)
                          if (sent_idx, word_idx) not in words_to_remove]

        if len(remaining_words) == 0:
            # Keep at least the first few words
            remaining_words = words[:min(3, len(words))]

        compressed_sentences.append(' '.join(remaining_words))

    result = ' '.join(compressed_sentences)

    # Capitalize first letter if needed
    if result and result[0].islower():
        result = result[0].upper() + result[1:]

    return result

def decompress_text(text):
    """
    Simple decompression - capitalize properly and clean up.
    Note: True decompression would require the LLM to reconstruct removed tokens.
    This is a basic formatting cleanup.
    """
    sentences = text.split('.')
    result = []

    for sentence in sentences:
        sentence = sentence.strip()
        if sentence:
            # Capitalize first letter
            sentence = sentence[0].upper() + sentence[1:]
            result.append(sentence)

    return '. '.join(result) + '.' if result else ''

def main():
    parser = argparse.ArgumentParser(
        description='MLM-based caveman compression using RoBERTa (no LLM API required)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Compress text with default settings (P=1e-5)
  python caveman_compress_mlm.py compress "Your text here"

  # Compress from file
  python caveman_compress_mlm.py compress -f input.txt

  # Conservative compression (less removal)
  python caveman_compress_mlm.py compress -f input.txt -p 1e-3

  # Aggressive compression (more removal)
  python caveman_compress_mlm.py compress -f input.txt -p 1e-6

  # Save to file
  python caveman_compress_mlm.py compress -f input.txt -o output.txt

  # Decompress
  python caveman_compress_mlm.py decompress "compressed text"

Probability thresholds (lower = more aggressive):
  P >= 1e-3:  Conservative (16% reduction, 98% accuracy)
  P >= 1e-4:  Moderate (19% reduction, 97% accuracy)
  P >= 1e-5:  Balanced (32% reduction, 92% accuracy) [DEFAULT]
  P >= 1e-6:  Aggressive (54% reduction, 83% accuracy)
        '''
    )

    parser.add_argument('mode', choices=['compress', 'decompress'],
                       help='compress or decompress')
    parser.add_argument('text', nargs='?', help='text to process')
    parser.add_argument('-f', '--file', help='input file path')
    parser.add_argument('-o', '--output', help='output file path')
    parser.add_argument('-p', '--prob-threshold', type=float, default=1e-5,
                       help='probability threshold: remove words with P >= this value (default: 1e-5)')
    parser.add_argument('--no-adjacent', action='store_true',
                       help='never remove two adjacent words; if both exceed threshold, remove only the highest probability one')
    parser.add_argument('--no-protect-ner', action='store_true',
                       help='disable NER protection (by default, named entities like PERSON, ORG, GPE, DATE are preserved)')

    args = parser.parse_args()

    # Get input text
    if args.file:
        input_path = Path(args.file)
        if not input_path.exists():
            print(f"Error: File not found: {args.file}", file=sys.stderr)
            sys.exit(1)
        input_text = input_path.read_text()
    elif args.text:
        input_text = args.text
    else:
        print("Error: Provide text or use -f for file input", file=sys.stderr)
        parser.print_help()
        sys.exit(1)

    # Process
    print("=" * 60)
    print(f"MODE: {args.mode.upper()} (MLM-BASED)")
    if args.mode == 'compress':
        print(f"THRESHOLD: P >= {args.prob_threshold:.0e}")
        if args.no_adjacent:
            print("NO-ADJACENT: enabled (won't remove consecutive words)")
        if not args.no_protect_ner:
            print("NER PROTECTION: enabled (named entities preserved)")
    print("=" * 60)
    print()

    if args.mode == 'compress':
        print("ORIGINAL TEXT:")
        print(input_text)
        print()
        print("Compressing...")
        print()

        output_text = compress_text(input_text, prob_threshold=args.prob_threshold,
                                    no_adjacent_removal=args.no_adjacent,
                                    protect_ner=not args.no_protect_ner)

        print("CAVEMAN COMPRESSED:")
        print(output_text)
        print()
        print("=" * 60)
        print("STATISTICS:")
        orig_tokens = count_tokens(input_text)
        comp_tokens = count_tokens(output_text)
        reduction = ((orig_tokens - comp_tokens) / orig_tokens * 100) if orig_tokens > 0 else 0
        print(f"  Original:    {len(input_text)} chars ≈ {orig_tokens} tokens")
        print(f"  Compressed:  {len(output_text)} chars ≈ {comp_tokens} tokens")
        print(f"  Reduction:  {reduction:.1f}%")
        print("=" * 60)

    else:  # decompress
        print("COMPRESSED TEXT:")
        print(input_text)
        print()
        print("Decompressing...")
        print()

        output_text = decompress_text(input_text)

        print("DECOMPRESSED:")
        print(output_text)
        print()

    # Save to file if requested
    if args.output:
        output_path = Path(args.output)
        output_path.write_text(output_text)
        print(f"\nSaved to: {args.output}")

if __name__ == '__main__':
    main()
