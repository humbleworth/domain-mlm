# Masking Strategy for Domain MLM

## Current Implementation

The masking is handled by HuggingFace's `DataCollatorForLanguageModeling` with these settings:

```python
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=True,
    mlm_probability=0.15,  # 15% of tokens are masked
)
```

## How It Works

### 1. Token-Level Masking (Current)
- **15% of tokens** are randomly selected for masking
- For BERT-like models, this means 15% of subword tokens
- For CANINE, this would be 15% of characters

### 2. Masking Process
When a token is selected for masking, one of three things happens:
- **80% of the time**: Replace with [MASK] token
- **10% of the time**: Replace with a random token
- **10% of the time**: Keep the original token

### 3. Example with Domains

For a domain like `example.com`:
- Tokenized (BERT): `['example', '.', 'com']` → might mask `['example', '[MASK]', 'com']`
- Tokenized (CANINE): `['e', 'x', 'a', 'm', 'p', 'l', 'e', '.', 'c', 'o', 'm']` → might mask `['e', '[MASK]', 'a', 'm', '[MASK]', 'l', 'e', '.', 'c', '[MASK]', 'm']`

## Domain-Specific Considerations

### Current Limitations
1. **Random masking** may not be optimal for domains
2. Might mask important parts like TLD (.com, .org)
3. No consideration for domain structure

### Potential Improvements

#### 1. Structured Masking
```python
def domain_aware_masking(domain):
    parts = domain.split('.')
    # Mask whole components: example.[MASK] or [MASK].com
    # Mask substrings: ex[MASK]ple.com
```

#### 2. Span Masking (CANINE-style)
```python
# Instead of individual characters, mask spans
# exam[MASK][MASK][MASK].com instead of e[MASK]a[MASK]p[MASK]e.com
```

#### 3. Component-Aware Masking
```python
# Higher probability to mask:
# - Subdomain (www, mail, etc.)
# - Domain name
# - Lower probability for TLD
```

## Customizing Masking

To implement custom masking, you would:

1. Create a custom data collator:
```python
class DomainDataCollatorForLanguageModeling(DataCollatorForLanguageModeling):
    def torch_mask_tokens(self, inputs, special_tokens_mask=None):
        # Custom masking logic here
        # Consider domain structure
        pass
```

2. Or preprocess domains before tokenization:
```python
def prepare_domain_for_mlm(domain):
    # Add special tokens to mark boundaries
    # <SUBDOMAIN>www</SUBDOMAIN><DOMAIN>example</DOMAIN><TLD>com</TLD>
    pass
```

## Current Configuration Options

You can adjust masking probability:
```bash
python train_mlm.py --mlm_probability 0.20  # Mask 20% instead of 15%
```

## Verification

To see actual masking in action:
```python
# In train_mlm.py, add after data_collator creation:
sample = tokenized_train[0]
masked_sample = data_collator([sample])
print(f"Original: {tokenizer.decode(sample['input_ids'])}")
print(f"Masked: {tokenizer.decode(masked_sample['input_ids'][0])}")
```