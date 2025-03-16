# Colorful CoNLL-U Format Visualization

This README demonstrates how to visualize CoNLL-U formatted text with colors using HTML and Markdown.

## 📌 Example CoNLL-U Data
```plaintext
# sent_id = 1
# text = This is an example sentence.
1	This	_	DET	_	_	_	_	_	_
2	is	_	VERB	_	_	_	_	_	_
3	an	_	DET	_	_	_	_	_	_
4	example	_	NOUN	_	_	_	_	_	_
5	sentence	_	NOUN	_	_	_	_	_	_
6	.	_	PUNCT	_	_	_	_	_	_
```

## 🎨 Colorful HTML Representation
You can use the following HTML snippet to visualize the text with colors:

```html
<pre>
<span style="color: green;"># sent_id = 1</span>
<span style="color: blue;"># text = This is an example sentence.</span>
<span style="color: red;">1</span> <span style="color: purple;">This</span> _ <span style="color: orange;">DET</span> _ _ _ _ _ _
<span style="color: red;">2</span> <span style="color: purple;">is</span> _ <span style="color: orange;">VERB</span> _ _ _ _ _ _
<span style="color: red;">3</span> <span style="color: purple;">an</span> _ <span style="color: orange;">DET</span> _ _ _ _ _ _
<span style="color: red;">4</span> <span style="color: purple;">example</span> _ <span style="color: orange;">NOUN</span> _ _ _ _ _ _
<span style="color: red;">5</span> <span style="color: purple;">sentence</span> _ <span style="color: orange;">NOUN</span> _ _ _ _ _ _
<span style="color: red;">6</span> <span style="color: purple;">.</span> _ <span style="color: orange;">PUNCT</span> _ _ _ _ _ _
</pre>
```

## 🖥️ Colorful Terminal Output (Python + ANSI Escape Codes)
If you want to display the colored output in a terminal, use the following Python script:

```python
def colorize(text, color_code):
    return f"\033[{color_code}m{text}\033[0m"

print(colorize("# sent_id = 1", "32"))  # Green
print(colorize("# text = This is an example sentence.", "34"))  # Blue

lines = [
    (1, "This", "DET"),
    (2, "is", "VERB"),
    (3, "an", "DET"),
    (4, "example", "NOUN"),
    (5, "sentence", "NOUN"),
    (6, ".", "PUNCT"),
]

for num, word, pos in lines:
    print(
        colorize(str(num), "31"),  # Red
        colorize(word, "35"),  # Purple
        "_",
        colorize(pos, "33"),  # Orange
        "_ _ _ _ _ _ _",
    )
```

This will display the CoNLL-U formatted text in a visually appealing way in the terminal.

## 🚀 Enjoy Your Colorful CoNLL-U Data!

