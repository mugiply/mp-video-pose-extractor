import { HandVector } from './hand-vector';
import { BodyVector } from './body-vector';
export interface PoseSetItem {
    id: number;
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
    extendedData?: {
        [key: string]: any;
    };
    mergedTimeMiliseconds?: number;
    mergedDurationMiliseconds?: number;
    debug?: {
        duplicatedItems: {
            timeMiliseconds: number;
            durationMiliseconds: number;
            bodySimilarity?: number;
            handSimilarity?: number;
        }[];
    };
}
