import { Injectable } from '@angular/core';
import { Pose } from '../classes/pose';

/**
 * ポーズを管理するためのサービス
 */
@Injectable({
  providedIn: 'root',
})
export class PoseComposerService {
  constructor() {}

  init(videoName: string): Pose {
    const pose = new Pose();
    pose.setVideoName(videoName);
    return pose;
  }

  async downloadAsJson(pose: Pose) {
    const blob = new Blob([await pose.getJson()], {
      type: 'application/json',
    });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.target = '_blank';
    a.download = `${pose.getVideoName()}-poses.json`;
    a.click();
  }

  async downloadAsZip(pose: Pose) {
    const content = await pose.getZip();
    const url = window.URL.createObjectURL(content);
    const a = document.createElement('a');
    a.href = url;
    a.target = '_blank';
    a.download = `${pose.getVideoName()}-poses.zip`;
    a.click();
  }
}
