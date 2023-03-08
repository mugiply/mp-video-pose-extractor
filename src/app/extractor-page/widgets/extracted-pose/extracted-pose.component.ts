import { Component, Input } from '@angular/core';
import { PoseSetItem } from 'ngx-mp-pose-extractor';
import { Clipboard } from '@angular/cdk/clipboard';
import { MatSnackBar } from '@angular/material/snack-bar';

@Component({
  selector: 'app-extracted-pose',
  templateUrl: './extracted-pose.component.html',
  styleUrls: ['../../../shared/shared.scss', './extracted-pose.component.scss'],
})
export class ExtractedPoseComponent {
  @Input()
  public previewImage: 'frame' | 'pose' | 'face' = 'frame';

  @Input()
  public poseSetItem: PoseSetItem;

  constructor(private clipboard: Clipboard, private snackBar: MatSnackBar) {}

  copyJsonToClipboard() {
    const item = { ...this.poseSetItem };
    delete item.frameImageDataUrl;
    delete item.poseImageDataUrl;
    delete item.faceFrameImageDataUrl;
    this.clipboard.copy(JSON.stringify(item, null, '\t'));

    this.snackBar.open(
      'クリップボードへ項目のJSONをコピーしました',
      undefined,
      {
        duration: 2000,
      }
    );
  }
}
