import { Injectable } from '@angular/core';
import { MatSnackBar } from '@angular/material/snack-bar';
import { POSE_LANDMARKS, Results } from '@mediapipe/holistic';
import * as JSZip from 'jszip';

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
    duration: number;
  };
  poses: PoseItem[];
  poseLandmarkMapppings: string[];
}

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

  start(videoName: string) {
    this.videoName = videoName;
    this.poses = [];
    this.jsZip = new JSZip();
  }

  getNumberOfPoses(): number {
    return this.poses.length;
  }

  push(
    videoTimeMiliseconds: number,
    videoFrameImageJpegDataUrl: string | undefined,
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

    if (videoFrameImageJpegDataUrl && this.jsZip) {
      try {
        const index =
          videoFrameImageJpegDataUrl.indexOf('base64,') + 'base64,'.length;
        videoFrameImageJpegDataUrl =
          videoFrameImageJpegDataUrl.substring(index);

        this.jsZip.file(
          `snapshot-${videoTimeMiliseconds}.jpg`,
          videoFrameImageJpegDataUrl,
          { base64: true }
        );
      } catch (error) {
        console.warn(
          `[PoseExporterService] push - Could not push frame image`,
          error
        );
      }
    }

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
