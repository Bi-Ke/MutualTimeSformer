"""
    Function: tool function.

    Author: Bike Chen
    Email: Bike.Chen@oulu.fi
    Date: April 2, 2021
"""


def compute_accuracy(preds, labels):
    return (preds == labels).mean()

