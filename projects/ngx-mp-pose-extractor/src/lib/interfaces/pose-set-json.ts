import { PoseSetJsonItem } from './pose-set-json-item';

export interface PoseSetJson {
  generator: string;
  version: number;
  video: {
    width: number;
    height: number;
    duration: number;
  };
  poses: PoseSetJsonItem[];
  poseLandmarkMapppings: string[];
}
