export interface PoseVector {
  rightWristToRightElbow: number[];
  rightElbowToRightShoulder: number[];
  leftWristToLeftElbow: number[];
  leftElbowToLeftShoulder: number[];
}

export interface PoseItem {
  t: number;
  pose: number[][];
  vectors: PoseVector;
  frameImageDataUrl?: string;
}

export interface PoseJson {
  generator: string;
  version: number;
  video: {
    width: number;
    height: number;
    duration: number;
  };
  poses: PoseItem[];
  poseLandmarkMapppings: string[];
}
