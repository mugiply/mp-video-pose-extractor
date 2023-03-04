import { HandVector } from './hand-vector';
import { BodyVector } from './body-vector';

export interface PoseSetItem {
  timeMiliseconds: number;
  durationMiliseconds: number;
  pose?: number[][];
  leftHand?: number[][];
  rightHand?: number[][];
  bodyVector: BodyVector;
  handVector?: HandVector;
  frameImageDataUrl?: string;
  poseImageDataUrl?: string;
  faceFrameImageDataUrl?: string;
  extendedData?: { [key: string]: any };
}
