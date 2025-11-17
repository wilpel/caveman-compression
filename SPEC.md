# Caveman Compression Specification v1.0

## Overview

This document defines the formal rules for Caveman Compression: a lossless semantic compression technique that removes linguistic overhead while preserving all information content.

**Design Goal**: Minimize token count while maintaining complete semantic fidelity.

**Core Principle**: Remove only what LLMs can deterministically reconstruct. LLMs excel at predicting grammar, connectives, and sentence structure. They cannot reliably predict facts, numbers, or constraints. We compress the former, preserve the latter.

---

## Core Rules

### Rule 1: Sentence Atomicity

Each sentence MUST express exactly one atomic thought.

**Formal Definition**: An atomic thought is a single fact, action, constraint, or logical step that cannot be meaningfully subdivided without creating sentence fragments.

```
❌ "Database needs index and query runs faster"
✅ "Database needs index. Query runs faster."
```

**Rationale**: Compound sentences introduce connectives ("and", "but", "or") that consume tokens.

---

### Rule 2: Word Count Limit

Sentences SHOULD contain 2-5 words.

**Exceptions allowed when**:
- Preserving a constraint requires 6-7 words
- Breaking further would lose semantic clarity
- Technical terms cannot be shortened without ambiguity

```
✅ "Need fast queries."                    (3 words)
✅ "Hash map offers O(1) lookup."          (5 words)
⚠️ "All cups must contain same number."   (6 words - acceptable for constraint)
❌ "We need to implement a fast query system that uses indexes."  (11 words - must break)
```

---

### Rule 3: Connective Elimination

Remove ALL logical connectives unless omission creates ambiguity.

**Prohibited words**:
- Causal: because, since, due to, owing to, as a result
- Contrastive: however, nevertheless, although, despite, but
- Sequential: therefore, thus, consequently, hence, then
- Purpose: in order to, so that, for the purpose of
- Conditional: if, unless (unless essential to logic)

**Replacement strategy**: Express cause-effect through sequential sentences.

```
❌ "Use index because query too slow"
✅ "Query too slow. Use index."

❌ "Although expensive, index improves speed"
✅ "Index costs space. Index improves speed."
```

---

### Rule 4: Active Voice Required

Use active voice. Avoid passive constructions.

```
❌ "The value is calculated by the function"
✅ "Function calculates value"

❌ "The database was optimized"
✅ "Optimized database"
```

---

### Rule 5: Voice and Tense

MUST use active voice and present tense unless temporal distinction is semantically critical.

**Active Voice Required**:
```
❌ "The value is calculated by the function"
✅ "Function calculates value"
```

**Present Tense Default**:
```
❌ "Will need to check constraints"
✅ "Need check constraints"

✅ "Was O(n) before. Now O(log n)."  (temporal distinction preserved)
```

---

### Rule 6: Specificity Requirement

Replace vague quantities with specific ones.

**Vague → Specific transformations**:

| ❌ Vague | ✅ Specific |
|------------|------------|
| "several variables" | "three variables" |
| "some constraints" | "two constraints" |
| "many requests" | "50 million requests" |
| "a few engineers" | "15 engineers" |

**Rationale**: Specific numbers preserve information, vague terms lose it.

```
❌ "The system handles a large number of daily requests"
✅ "System handles 50 million daily requests"
```

---

### Rule 7: Modifier Minimization

Remove adjectives and adverbs UNLESS they carry distinct semantic information.

**Semantic test**: If removing the modifier changes truth value or adds ambiguity, keep it. Otherwise, remove.

```
❌ "very important constraint"    → ✅ "constraint"
❌ "quickly process data"         → ✅ "process data"
❌ "extremely efficient algorithm"→ ✅ "efficient algorithm" or "O(log n) algorithm"

✅ "key constraint" (if distinguishing from non-key)
✅ "same number" (same vs. different carries meaning)
```

---

### Rule 8: Article Omission

Omit articles (`a`, `an`, `the`) when context provides sufficient specificity.

```
❌ "The database needs an index"
✅ "Database needs index"

❌ "Add the value to the array"
✅ "Add value to array"
```

**Exception**: Keep articles when omission creates ambiguity between generic and specific references.

---

### Rule 9: Pronoun Handling

Replace pronouns with explicit nouns when:
- Antecedent is not in immediately previous sentence
- Multiple possible referents exist
- Pronoun is more than 3 letters ("it" OK, "they" OK, "their" → replace)

```
❌ "Function returns value. Store it in variable. Use it later."
✅ "Function returns value. Store value in variable. Use value later."
```

---

### Rule 10: Logical Completeness

Every inference step MUST be explicit. No implicit logical leaps allowed.

**Test**: Can a reader reconstruct the full reasoning chain from the compressed text?

```
❌ "Need hash map. Use hash map."
✅ "Need fast lookup. Array too slow. Hash map offers O(1). Use hash map."
   (Shows: need → problem → solution → decision)

❌ "Bug in code. Fix bug."
✅ "Bug in code. Bug causes crash. Find crash location. Fix null check. Bug fixed."
   (Shows: problem → symptom → investigation → solution → verification)
```

---

## Edge Cases

### Conditionals

**Simple conditionals**: Omit "if" when condition-action pair is clear.

```
❌ "If value greater than ten, return error"
✅ "Value greater than ten. Return error."
```

**Complex conditionals**: Keep "if" when multiple interleaved conditions exist.

```
✅ "If A and B then do X. If A and not B then do Y."
   (Removing 'if' creates ambiguity)
```

---

### Lists and Enumerations

Keep collective references when already concise.

```
✅ "Check all connections"
✅ "Verify each service"
✅ "Test three endpoints"
```

Only enumerate when specificity adds information:

```
❌ "Install dependencies A, B, and C"
✅ "Install dependencies: React, Node, PostgreSQL"
```

---

### Technical Terminology

**Preserve technical terms** that have precise meanings. Do not "simplify" domain-specific vocabulary.

```
✅ "Use binary search"    (not "use fast search")
✅ "Implement recursion"  (not "function calls function")
✅ "Check Big-O"          (not "check speed")
```

---

### Negative Statements

**Use explicit negation** rather than implicit absence.

```
❌ "No error found"          → ✅ "Error not found"
❌ "Function doesn't return" → ✅ "Function returns nothing"
```

---

## Validation Algorithm

To verify Caveman Compression correctness:

1. **Extract facts**: List all discrete facts from original text
2. **Extract facts from compressed**: List all discrete facts from compressed text
3. **Compare sets**: Facts must be identical
4. **Check logical flow**: Verify reasoning chain has no gaps
5. **Measure compression**: Calculate token reduction percentage

**Acceptance criteria**:
- Fact preservation: Complete
- Logical completeness: Complete
- Token reduction: ≥30%

---

## Anti-Patterns

### Anti-Pattern 1: Telegraphic Ambiguity

```
❌ "Function error return null"
✅ "Function has error. Function returns null."
```

**Problem**: Word order unclear; relation between words ambiguous.

---

### Anti-Pattern 2: Over-Compression

```
❌ "Try fix"
✅ "Try option A. Measure result. Pick best option. Fix applied."
```

**Problem**: Skips intermediate logical steps.

---

### Anti-Pattern 3: Information Addition

```
❌ Original: "Use hash map"
❌ Caveman: "Use hash map. Hash map is fast."

✅ Caveman: "Use hash map."
```

**Problem**: Added information not in original ("is fast").

---

### Anti-Pattern 4: False Concreteness

```
❌ "Test five to eight" → "Test five through eight"
✅ "Test five to eight" → "Test five. Test six. Test seven. Test eight."
```

**Problem**: "Through" is still abstract; must enumerate.

---

## Conformance Levels

### Level 1: Basic Caveman
- Rules 1-5 applied
- 15-20% token reduction typical
- Sufficient for most use cases

### Level 2: Strict Caveman
- Rules 1-8 applied
- 20-30% token reduction typical
- Recommended for token-constrained contexts

### Level 3: Maximum Caveman
- All rules applied rigorously
- 30-40% token reduction typical
- Use only when token cost is critical

---

## Examples by Category

### Algorithm Explanation

**Original (61 tokens)**:
```
Binary search works by repeatedly dividing the search space in half. First,
compare the target value to the middle element. If they match, return the
position. If target is less, search the left half. If target is greater,
search the right half.
```

**Level 2 Caveman (49 tokens)**:
```
Binary search divides search space in half. Compare target to middle element.
Match found. Return position. Target less than middle. Search left half.
Target greater than middle. Search right half.
```

**Reduction**: 20%

---

### Debugging Steps

**Original (66 tokens)**:
```
When debugging the null pointer exception, start by checking the stack trace
to identify where the error occurred. Then examine the code at that line to
see which variable is null. Add a null check before the dereference to
prevent the error from happening again.
```

**Level 2 Caveman (54 tokens)**:
```
Debugging null pointer exception. Check stack trace. Find error location.
Examine code at error line. Identify null variable. Variable needs value
check. Add null check before dereference. Prevents error next time.
```

**Reduction**: 18%

---

## Version History

- **v1.0** (2025-01): Initial specification release

---

## References

- Token counting based on GPT tokenizer conventions
- Semantic preservation inspired by information theory (Shannon entropy)
- Influenced by controlled natural languages (Simplified English, ASD-STE100)
