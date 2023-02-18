import { HandVector } from './hand-vector';
import { BodyVector } from './body-vector';
export interface PoseSetItem {
    timeMiliseconds: number;
    durationMiliseconds: number;
    pose?: number[][];
    leftHand?: number[][];
    rightHand?: number[][];
    bodyVectors: BodyVector;
    handVectors?: HandVector;
    frameImageDataUrl?: string;
    poseImageDataUrl?: string;
}
