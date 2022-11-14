import numpy as np

class Filter:

    def __intersection(self, rect1, rect2):
        """
        Calculates square of intersection of two rectangles
        rect: list with coords of top-right and left-boom corners [x1,y1,x2,y2]
        return: square of intersection
        """
        x_overlap = max(0, min(rect1[2], rect2[2]) - max(rect1[0], rect2[0]))
        y_overlap = max(0, min(rect1[3], rect2[3]) - max(rect1[1], rect2[1]))
        overlapArea = x_overlap * y_overlap
        return overlapArea

    def __square(self, rect):
        return abs(rect[2] - rect[0]) * abs(rect[3] - rect[1])
    
    def __nms(self, rects, thd=0.5):
        """
        Filter rectangles
        rects is array of oblects ([x1,y1,x2,y2], confidence, class)
        thd - intersection threshold (intersection divides min square of rectange)
        """
        out = []

        remove = [False] * len(rects)

        for i in range(0, len(rects) - 1):
            if remove[i]:
                continue
            inter = [0.0] * len(rects)
            for j in range(i, len(rects)):
                if remove[j]:
                    continue
                inter[j] = self.__intersection(rects[i][0], rects[j][0]) / min(self.__square(rects[i][0]), self.__square(rects[j][0]))

            max_prob = 0.0
            max_idx = 0
            for k in range(i, len(rects)):
                if inter[k] >= thd:
                    if rects[k][1] > max_prob:
                        max_prob = rects[k][1]
                        max_idx = k

            for k in range(i, len(rects)):
                if (inter[k] >= thd) & (k != max_idx):
                    remove[k] = True

        for k in range(0, len(rects)):
            if not remove[k]:
                out.append(rects[k])

        boxes = [box[0] for box in out]
        scores = [score[1] for score in out]
        classes = [cls[2] for cls in out]
        return boxes, scores, classes
    
    def	intersection(self, detections, th=0.5):
        # creating a zip object that will contain model output info as
        output_info = list(zip(detections['detection_boxes'], detections['detection_scores'], detections['detection_classes'] ))
        boxes, scores, classes = self.__nms(output_info)
        detections['detection_boxes'] = np.array(boxes)
        detections['detection_scores'] = np.array(scores)
        detections['detection_classes'] = np.array(classes, dtype=np.int64)

        return detections

    def classes(self, detections, classes_to_find, keys_of_interest =['detection_classes', 'detection_boxes', 'detection_scores']):
        detections = {key: value for key, value in detections.items() if key in keys_of_interest}
        classes = detections['detection_classes']
        for key in keys_of_interest:
            current_array = detections[key]
            filtered_current_array = []
            for class_to_find in classes_to_find:
                filtered_current_array = current_array[classes == class_to_find]
            detections[key] = filtered_current_array

        return detections
    
    def score(self, detections, min_score=0.25, keys_of_interest =['detection_classes', 'detection_boxes', 'detection_scores']):
        detections = {key: value for key, value in detections.items() if key in keys_of_interest}
        
        for key in keys_of_interest:
            scores = detections['detection_scores']
            current_array = detections[key]
            filtered_current_array = current_array[scores > min_score]
            detections[key] = filtered_current_array

        return detections

if __name__ == '__main__':
    print('Opss... Wrong way.')
    print('This is class is not acessable directly... you should turn back.')