import { PoseItem } from './pose-item';

export interface SimilarPoseItem extends PoseItem {
  similarity: number;
  bodySimilarity: number;
  handSimilarity?: number;
}
