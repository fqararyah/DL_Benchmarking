def convert_eval_entries_to_kenning_eval(dataset, cocoeval):
    """
    Converts predictions for the object detection task
    to format applicable for evaluation plots.
    
    Parameters
    ----------
    dataset : COCODataset2017
        Dataset object with data handling and evaluation metrics
    cocoeval : List[Dict]
        Prediction entries for COCO dataset evaluation
    """
    # Firstly, let's group prediction entries by images they belong to.
    # COCODataset2017, as well as other object detection datasets in
    # Kenning use DectObject objects as representation of entries.
    grouped_entries = defaultdict(list)
    for entry in cocoeval:
        width = dataset.coco.imgs[entry['image_id']]['width']
        height = dataset.coco.imgs[entry['image_id']]['height']
        bbox = entry['bbox']
        
        grouped_entries[entry['image_id']].append(
            DectObject(
                clsname=dataset.classnames[entry['category_id']],
                xmin=bbox[0] / width,
                ymin=bbox[1] / height,
                xmax=(bbox[0] + bbox[2]) / width,
                ymax=(bbox[1] + bbox[3]) / height,
                score=entry['score']
            )
        )
    # After grouping predictions, and converting them to DectObjects,
    # we create a Measurements object - it is a bag for various quality
    # and performance metrics used throughout Kenning.
    # Dataset's evaluate method use it to store evaluation results.
    measurements = Measurements()
    
    # from this point, we iterate over image IDs and ground truth
    # to evaluate predictions.
    for X, y in zip(dataset.dataX, dataset.dataY):
        measurements += dataset.evaluate(
            [grouped_entries[X]],
            dataset.prepare_output_samples([y])
        )
    # in the end, we additionally save the class names
    measurements += {'class_names': dataset.classnames}
    return measurements.data