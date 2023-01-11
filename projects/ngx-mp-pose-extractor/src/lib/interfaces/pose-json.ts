import { PoseJsonItem } from './pose-json-item';

export interface PoseJson {
  generator: string;
  generatedVersion: number;
  version: number;
  video: {
    width: number;
    height: number;
    duration: number;
  };
  poses: PoseJsonItem[];
  poseLandmarkMapppings: string[];
}
