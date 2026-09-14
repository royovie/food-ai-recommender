from backend.prediction_utils import determine_prediction

def test_high_confidence_prediction():
    predictions = [
        {"food": "pizza", "confidence": 80},
        {"food": "lasagna", "confidence": 40},
    ]

    result = determine_prediction(predictions)

    assert result == "pizza"


def test_low_confidence_prediction():
    predictions = [
        {"food": "pizza", "confidence": 45},
        {"food": "lasagna", "confidence": 20},
    ]

    result = determine_prediction(predictions)

    assert result == "uncertain"


def test_small_confidence_gap():
    predictions = [
        {"food": "pizza", "confidence": 70},
        {"food": "lasagna", "confidence": 62},
    ]

    result = determine_prediction(predictions)

    assert result == "uncertain"