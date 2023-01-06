import { Injectable } from '@angular/core';
import { MatSnackBar } from '@angular/material/snack-bar';
import { POSE_LANDMARKS, Results } from '@mediapipe/holistic';
import * as JSZip from 'jszip';
import { PoseItem, PoseJson } from './pose';

// @ts-ignore
const cosSimilarity = require('cos-similarity');

@Injectable({
  providedIn: 'root',
})
export class PoseExporterService {
  private videoName?: string;
  private videoMetadata?: {
    width: number;
    height: number;
    duration: number;
  };

  private poses: PoseItem[] = [];
  private jsZip?: JSZip;

  constructor(private snackBar: MatSnackBar) {}

  init(videoName: string) {
    this.videoName = videoName;
    this.poses = [];
    this.jsZip = new JSZip();
  }

  getNumberOfPoses(): number {
    return this.poses.length;
  }

  getPoses(): PoseItem[] {
    return this.poses;
  }

  push(
    videoTimeMiliseconds: number,
    frameImageJpegDataUrl: string | undefined,
    videoWidth: number,
    videoHeight: number,
    videoDuration: number,
    results: Results
  ) {
    this.videoMetadata = {
      width: videoWidth,
      height: videoHeight,
      duration: videoDuration,
    };

    if (results.poseLandmarks === undefined) return;

    const poseLandmarksWithWorldCoordinate: any[] = (results as any).ea
      ? (results as any).ea
      : [];
    if (poseLandmarksWithWorldCoordinate.length === 0) {
      console.warn(
        `[PoseExporterService] push - Could not get the pose with the world coordinate`,
        results
      );
      return;
    }

    const vectors = {
      rightWristToRightElbow: [
        poseLandmarksWithWorldCoordinate[POSE_LANDMARKS.RIGHT_WRIST].x -
          poseLandmarksWithWorldCoordinate[POSE_LANDMARKS.RIGHT_ELBOW].x,
        poseLandmarksWithWorldCoordinate[POSE_LANDMARKS.RIGHT_WRIST].y -
          poseLandmarksWithWorldCoordinate[POSE_LANDMARKS.RIGHT_ELBOW].y,
        poseLandmarksWithWorldCoordinate[POSE_LANDMARKS.RIGHT_WRIST].z -
          poseLandmarksWithWorldCoordinate[POSE_LANDMARKS.RIGHT_ELBOW].z,
      ],
      rightElbowToRightShoulder: [
        poseLandmarksWithWorldCoordinate[POSE_LANDMARKS.RIGHT_ELBOW].x -
          poseLandmarksWithWorldCoordinate[POSE_LANDMARKS.RIGHT_SHOULDER].x,
        poseLandmarksWithWorldCoordinate[POSE_LANDMARKS.RIGHT_ELBOW].y -
          poseLandmarksWithWorldCoordinate[POSE_LANDMARKS.RIGHT_SHOULDER].y,
        poseLandmarksWithWorldCoordinate[POSE_LANDMARKS.RIGHT_ELBOW].z -
          poseLandmarksWithWorldCoordinate[POSE_LANDMARKS.RIGHT_SHOULDER].z,
      ],
      leftWristToLeftElbow: [
        poseLandmarksWithWorldCoordinate[POSE_LANDMARKS.LEFT_WRIST].x -
          poseLandmarksWithWorldCoordinate[POSE_LANDMARKS.LEFT_ELBOW].x,
        poseLandmarksWithWorldCoordinate[POSE_LANDMARKS.LEFT_WRIST].y -
          poseLandmarksWithWorldCoordinate[POSE_LANDMARKS.LEFT_ELBOW].y,
        poseLandmarksWithWorldCoordinate[POSE_LANDMARKS.LEFT_WRIST].z -
          poseLandmarksWithWorldCoordinate[POSE_LANDMARKS.LEFT_ELBOW].z,
      ],
      leftElbowToLeftShoulder: [
        poseLandmarksWithWorldCoordinate[POSE_LANDMARKS.LEFT_ELBOW].x -
          poseLandmarksWithWorldCoordinate[POSE_LANDMARKS.LEFT_SHOULDER].x,
        poseLandmarksWithWorldCoordinate[POSE_LANDMARKS.LEFT_ELBOW].y -
          poseLandmarksWithWorldCoordinate[POSE_LANDMARKS.LEFT_SHOULDER].y,
        poseLandmarksWithWorldCoordinate[POSE_LANDMARKS.LEFT_ELBOW].z -
          poseLandmarksWithWorldCoordinate[POSE_LANDMARKS.LEFT_SHOULDER].z,
      ],
    };

    const pose: PoseItem = {
      t: videoTimeMiliseconds,
      pose: poseLandmarksWithWorldCoordinate.map((landmark) => {
        return [landmark.x, landmark.y, landmark.z, landmark.visibility];
      }),
      vectors: vectors,
      frameImageDataUrl: frameImageJpegDataUrl,
    };

    if (1 <= this.poses.length) {
      const lastPose = this.poses[this.poses.length - 1];
      if (this.isSimilarPose(lastPose, pose)) {
        return;
      }
    }

    this.poses.push(pose);
  }

  isSimilarPose(pose1: PoseItem, pose2: PoseItem, threshold = 0.9): boolean {
    const cosSimilarities = {
      leftWristToLeftElbow: cosSimilarity(
        pose1.vectors.leftWristToLeftElbow,
        pose2.vectors.leftWristToLeftElbow
      ),
      leftElbowToLeftShoulder: cosSimilarity(
        pose1.vectors.leftElbowToLeftShoulder,
        pose2.vectors.leftElbowToLeftShoulder
      ),
      rightWristToRightElbow: cosSimilarity(
        pose1.vectors.rightWristToRightElbow,
        pose2.vectors.rightWristToRightElbow
      ),
      rightElbowToRightShoulder: cosSimilarity(
        pose1.vectors.rightElbowToRightShoulder,
        pose2.vectors.rightElbowToRightShoulder
      ),
    };

    let isSimilar = false;
    const cosSimilaritiesSum = Object.values(cosSimilarities).reduce(
      (sum, value) => sum + value,
      0
    );
    if (cosSimilaritiesSum >= threshold * Object.keys(cosSimilarities).length)
      isSimilar = true;

    console.log(
      `[PoseExporterService] isSimilarPose (${pose1.t} <-> ${pose2.t})`,
      isSimilar,
      cosSimilarities
    );

    return isSimilar;
  }

  removeDuplicatedPoses() {
    const newPoses: PoseItem[] = [];
    let lastPose: PoseItem | undefined = undefined;
    for (const pose of this.poses) {
      if (lastPose === undefined) {
        lastPose = pose;
        newPoses.push(pose);
        continue;
      }

      if (pose.t === lastPose.t) continue;
      lastPose = pose;
      newPoses.push(pose);
    }
  }

  downloadAsJson() {
    const message = this.snackBar.open(
      '保存するデータを生成しています... しばらくお待ちください...'
    );

    const blob = new Blob([this.getJson()], {
      type: 'application/json',
    });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.target = '_blank';
    a.download = `${this.videoName}-poses.json`;
    a.click();

    message.dismiss();
  }

  async downloadAsZip() {
    if (!this.jsZip) return;

    const message = this.snackBar.open(
      '保存するデータを生成しています... しばらくお待ちください...'
    );

    this.jsZip.file('poses.json', this.getJson());

    for (const pose of this.poses) {
      if (!pose.frameImageDataUrl) continue;
      try {
        const index =
          pose.frameImageDataUrl.indexOf('base64,') + 'base64,'.length;
        const base64 = pose.frameImageDataUrl.substring(index);

        this.jsZip.file(`snapshot-${pose.t}.jpg`, base64, { base64: true });
      } catch (error) {
        console.warn(
          `[PoseExporterService] push - Could not push frame image`,
          error
        );
      }
    }

    const content = await this.jsZip.generateAsync({ type: 'blob' });
    const url = window.URL.createObjectURL(content);
    const a = document.createElement('a');
    a.href = url;
    a.target = '_blank';
    a.download = `${this.videoName}-poses.zip`;
    a.click();

    message.dismiss();
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
      video: this.videoMetadata!,
      poses: this.poses,
      poseLandmarkMapppings: poseLandmarkMappings,
    };

    return JSON.stringify(json);
  }
}
