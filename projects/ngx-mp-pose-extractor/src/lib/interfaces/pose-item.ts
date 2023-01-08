import { PoseVector } from './pose-vector';

export interface PoseItem {
  timeMiliseconds: number;
  durationMiliseconds: number;
  pose?: number[][];
  vectors: PoseVector;
  frameImageDataUrl?: string;
  poseImageDataUrl?: string;
}
