import { Component, OnInit } from '@angular/core';
import { PoseComposerService } from 'projects/ngx-mp-pose-extractor/src/lib/services/pose-composer.service';

@Component({
  selector: 'app-tester-page',
  templateUrl: './tester-page.component.html',
  styleUrls: ['../shared/shared.scss', './tester-page.component.scss'],
  providers: [PoseComposerService],
})
export class TesterPageComponent implements OnInit {
  public poseFileName?: string;
  public poseFileType?: 'zip' | 'json';
  public poseZipArrayBuffer?: ArrayBuffer;
  public poseJson?: string;

  constructor() {}

  ngOnInit() {}

  public ohChoosePoseFile(event: any) {
    const files: File[] = event.target.files;
    if (files.length === 0) return;

    const file = files[0];

    this.poseFileName = file.name;
    this.poseFileType = file.type === 'application/zip' ? 'zip' : 'json';

    const reader = new FileReader();
    reader.onload = async (event) => {
      const result = event.target?.result;
      if (this.poseFileType === 'zip') {
        this.poseZipArrayBuffer = result as ArrayBuffer;
      } else {
        this.poseJson = result as string;
      }
    };

    if (this.poseFileType === 'zip') {
      reader.readAsArrayBuffer(file);
    } else {
      reader.readAsText(file);
    }
  }
}
