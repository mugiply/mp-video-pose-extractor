import { Injectable } from '@angular/core';
import { PoseSet } from '../classes/pose-set';

/**
 * ポーズを管理するためのサービス
 */
@Injectable({
  providedIn: 'root',
})
export class PoseComposerService {
  constructor() {}

  init(videoName: string): PoseSet {
    const poseSet = new PoseSet();
    poseSet.setVideoName(videoName);
    return poseSet;
  }

  async downloadAsJson(poseSet: PoseSet) {
    const blob = new Blob([await poseSet.getJson()], {
      type: 'application/json',
    });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.target = '_blank';
    a.download = `${poseSet.getVideoName()}.poses.json`;
    a.click();
  }

  async downloadAsZip(poseSet: PoseSet) {
    const content = await poseSet.getZip();
    const url = window.URL.createObjectURL(content);
    const a = document.createElement('a');
    a.href = url;
    a.target = '_blank';
    a.download = `${poseSet.getVideoName()}.poses.zip`;
    a.click();
  }
}
