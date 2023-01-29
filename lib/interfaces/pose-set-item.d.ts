import { PoseVector } from './pose-vector';
export interface PoseSetItem {
    timeMiliseconds: number;
    durationMiliseconds: number;
    pose?: number[][];
    vectors: PoseVector;
    frameImageDataUrl?: string;
    poseImageDataUrl?: string;
}
