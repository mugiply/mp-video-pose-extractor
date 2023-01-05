import { TestBed } from '@angular/core/testing';

import { PoseExporterService } from './pose-exporter.service';

describe('PoseExporterService', () => {
  let service: PoseExporterService;

  beforeEach(() => {
    TestBed.configureTestingModule({});
    service = TestBed.inject(PoseExporterService);
  });

  it('should be created', () => {
    expect(service).toBeTruthy();
  });
});
