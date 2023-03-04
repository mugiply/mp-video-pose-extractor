import { PoseSetItem } from './pose-set-item';

export interface SimilarPoseItem extends PoseSetItem {
  similarity: number;
  bodyPoseSimilarity?: number;
  handPoseSimilarity?: number;
}
