from typing import Text

def is_float(text: Text):
    try:
        x = float(text)
        return True
    except:
        return False