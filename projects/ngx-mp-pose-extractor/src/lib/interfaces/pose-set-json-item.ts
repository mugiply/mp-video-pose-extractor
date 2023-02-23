export interface PoseSetJsonItem {
  // timeMiliseconds
  t: number;
  // durationMiliseconds
  d: number;
  // pose
  p?: number[][];
  // leftHand
  l?: number[][];
  // rightHand
  r?: number[][];
  // bodyVectors
  v: number[][];
  // handVectors
  h?: (number[] | null)[];
  // extendedData
  e?: { [key: string]: any };
}
