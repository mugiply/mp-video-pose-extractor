import { ComponentFixture, TestBed } from '@angular/core/testing';

import { ExtractedPoseComponent } from './extracted-pose.component';

describe('ExtractedPoseComponent', () => {
  let component: ExtractedPoseComponent;
  let fixture: ComponentFixture<ExtractedPoseComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      declarations: [ ExtractedPoseComponent ]
    })
    .compileComponents();

    fixture = TestBed.createComponent(ExtractedPoseComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
