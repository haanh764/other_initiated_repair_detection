import argparse

def int_or_max(value):
    if value == "max":
        return value
    try:
        return int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"'{value}' is not a valid integer or 'max'")
