export interface ExtendedClassifier {
  name: string;
  url: string;
  target: 'frameImage' | 'faceFrameImage';
}

export type ExtendedClassifierData = {
  [key: string]: {
    label: string;
    prob: number;
  }[];
};
