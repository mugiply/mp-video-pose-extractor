import { PoseVector } from './pose-vector';

export interface PoseItem {
  t: number;
  pose?: number[][];
  vectors: PoseVector;
  frameImageDataUrl?: string;
  poseImageDataUrl?: string;
}
