import { ComponentFixture, TestBed } from '@angular/core/testing';

import { ExtractorComponent } from './extractor.component';

describe('ExtractorComponent', () => {
  let component: ExtractorComponent;
  let fixture: ComponentFixture<ExtractorComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      declarations: [ ExtractorComponent ]
    })
    .compileComponents();

    fixture = TestBed.createComponent(ExtractorComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
