import { Injectable } from '@angular/core';
import { POSE_LANDMARKS, Results } from '@mediapipe/holistic';

interface PoseItem {
  t: number;
  pose: number[][];
  rHand?: number[][];
  lHand?: number[][];
}

interface PoseJson {
  generator: string;
  version: number;
  video: {
    width: number;
    height: number;
  };
  poses: PoseItem[];
  poseLandmarkMapppings: string[];
}

@Injectable({
  providedIn: 'root',
})
export class PoseExporterService {
  private videoName?: string;
  private poses: PoseItem[] = [];
  private videoWidth: number = 0;
  private videoHeight: number = 0;

  constructor() {
    console.log();
  }

  downloadAsJson() {
    const blob = new Blob([this.getJson()], {
      type: 'application/json',
    });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.target = '_blank';
    a.download = `${this.videoName}-poses.json`;
    a.click();
  }

  start(videoName: string) {
    this.videoName = videoName;
    this.poses = [];
  }

  getNumberOfPoses(): number {
    return this.poses.length;
  }

  push(
    videoTimeMiliseconds: number,
    videoWidth: number,
    videoHeight: number,
    results: Results
  ) {
    this.videoWidth = videoWidth;
    this.videoHeight = videoHeight;

    if (results.poseLandmarks === undefined) return;

    const pose: PoseItem = {
      t: videoTimeMiliseconds,
      pose: results.poseLandmarks.map((landmark) => {
        return [landmark.x, landmark.y, landmark.z];
      }),
    };

    if (results.rightHandLandmarks) {
      pose.rHand = results.rightHandLandmarks.map((landmark) => {
        return [landmark.x, landmark.y, landmark.z];
      });
    }

    if (results.leftHandLandmarks) {
      pose.lHand = results.leftHandLandmarks.map((landmark) => {
        return [landmark.x, landmark.y, landmark.z];
      });
    }

    this.poses.push(pose);
  }

  private getJson(): string {
    if (this.videoName === undefined) return '{}';

    let poseLandmarkMappings = [];
    for (const key of Object.keys(POSE_LANDMARKS)) {
      const index: number = POSE_LANDMARKS[key as keyof typeof POSE_LANDMARKS];
      poseLandmarkMappings[index] = key;
    }

    const json: PoseJson = {
      generator: 'mp-video-pose-extractor',
      version: 1,
      video: {
        width: this.videoWidth,
        height: this.videoHeight,
      },
      poses: this.poses,
      poseLandmarkMapppings: poseLandmarkMappings,
    };

    return JSON.stringify(json);
  }
}
