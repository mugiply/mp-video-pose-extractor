import { Injectable } from '@angular/core';
import * as tf from '@tensorflow/tfjs';
import * as tmImage from '@teachablemachine/image';

import { PoseSetItem } from 'projects/ngx-mp-pose-extractor/src/public-api';
import {
  ExtendedClassifier as ExtendedClassifierDefinition,
  ExtendedClassifierData,
} from '../interfaces/extended-classifier';
import { MatSnackBar } from '@angular/material/snack-bar';

type Classifier = ExtendedClassifierDefinition & {
  model: tmImage.CustomMobileNet;
  classLabels: string[];
};

@Injectable({
  providedIn: 'root',
})
export class ExtendedClassifierService {
  private classifiers: Classifier[] = [];

  constructor(private snackBar: MatSnackBar) {}

  async init(classifierDefinitions: ExtendedClassifierDefinition[]) {
    const classifiers = [];
    const now = Date.now();
    for (const classifierDefinition of classifierDefinitions) {
      const model = await tmImage.load(
        classifierDefinition.url + '/model.json' + '?t=' + now,
        classifierDefinition.url + '/metadata.json?t=' + now
      );
      classifiers.push({
        ...classifierDefinition,
        model: model,
        classLabels: model.getClassLabels(),
      });
    }
    this.classifiers = classifiers;
    console.log(`[ExtendedClassifierService] init - classifiers:`, classifiers);
  }

  async classify(pose: PoseSetItem): Promise<ExtendedClassifierData> {
    let frameImage;
    if (pose.frameImageDataUrl) {
      frameImage = new Image();
      frameImage.src = pose.frameImageDataUrl;
      await frameImage.decode();
    }

    let faceFrameImage;
    if (pose.faceFrameImageDataUrl) {
      faceFrameImage = new Image();
      faceFrameImage.src = pose.faceFrameImageDataUrl;
      await faceFrameImage.decode();
    }

    const result: ExtendedClassifierData = {};
    for (const classifier of this.classifiers) {
      let image;
      if (classifier.target === 'frameImage' && frameImage) {
        image = frameImage;
      } else if (classifier.target === 'faceFrameImage' && faceFrameImage) {
        image = faceFrameImage;
      }

      if (!image) {
        console.warn(
          `[ExtendedClassifierService] classify - No image for ${classifier.name}.`,
          classifier
        );
        continue;
      }

      const prediction = await classifier.model.predict(image);
      let classified;
      for (const pred of prediction) {
        if (!classified || classified.probability < pred.probability) {
          classified = pred;
        }
      }
      if (!classified) {
        console.warn(
          `[ExtendedClassifierService] classify - No prediction for ${classifier.name}.`,
          classifier
        );
        continue;
      }

      const sortedPrediction = prediction.sort((a, b) => {
        return b.probability - a.probability;
      });
      result[classifier.name] = sortedPrediction.map((pred) => {
        return {
          label: pred.className,
          prob: pred.probability,
        };
      });
    }

    return result;
  }
}
