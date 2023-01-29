import { Results } from '@mediapipe/holistic';
import { PoseItem } from '../interfaces/pose-item';
import { PoseVector } from '../interfaces/pose-vector';
import { SimilarPoseItem } from '../interfaces/matched-pose-item';
export declare class Pose {
    generator?: string;
    version?: number;
    private videoMetadata;
    poses: PoseItem[];
    isFinalized?: boolean;
    static readonly IS_ENABLE_DUPLICATED_POSE_REDUCTION = true;
    static readonly POSE_VECTOR_MAPPINGS: string[];
    private readonly IMAGE_JPEG_QUALITY;
    private readonly IMAGE_WIDTH;
    constructor();
    getVideoName(): string;
    setVideoName(videoName: string): void;
    setVideoMetaData(width: number, height: number, duration: number): void;
    getNumberOfPoses(): number;
    getPoses(): PoseItem[];
    getPoseByTime(timeMiliseconds: number): PoseItem | undefined;
    pushPose(videoTimeMiliseconds: number, frameImageJpegDataUrl: string | undefined, poseImageJpegDataUrl: string | undefined, videoWidth: number, videoHeight: number, videoDuration: number, results: Results): void;
    finalize(): Promise<void>;
    getSimilarPoses(results: Results, threshold?: number): SimilarPoseItem[];
    static getPoseVector(poseLandmarks: {
        x: number;
        y: number;
        z: number;
    }[]): PoseVector | undefined;
    static isSimilarPose(poseVectorA: PoseVector, poseVectorB: PoseVector, threshold?: number): boolean;
    static getPoseSimilarity(poseVectorA: PoseVector, poseVectorB: PoseVector): number;
    getZip(): Promise<Blob>;
    getJson(): Promise<string>;
    loadJson(json: string | any): void;
    loadZip(buffer: ArrayBuffer, includeImages?: boolean): Promise<void>;
}
