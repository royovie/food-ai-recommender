def determine_prediction(top_predictions):
    top1_confidence = top_predictions[0]["confidence"]
    top2_confidence = top_predictions[1]["confidence"]

    confidence_gap = top1_confidence - top2_confidence

    if top1_confidence < 50 or confidence_gap < 15:
        return "uncertain"

    return top_predictions[0]["food"]